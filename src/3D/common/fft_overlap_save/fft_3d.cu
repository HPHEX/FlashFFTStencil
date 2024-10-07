#include "fft_3d.cuh"
#include <cmath>
#include <vector>
#include <mutex>
#include <new>
#include <limits>
namespace flashfft3d_detail {
__device__ double2 fft_twiddle[64];
__device__ int tables_ready=0;
__device__ __forceinline__ double2 addz(double2 a,double2 b){return make_double2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ double2 subz(double2 a,double2 b){return make_double2(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ double2 shflz(double2 a,int delta,int width){return make_double2(__shfl_xor_sync(0xffffffff,a.x,delta,width),__shfl_xor_sync(0xffffffff,a.y,delta,width));}
template<int N,int S,bool Inverse,bool Simple=false>
__device__ __forceinline__ double2 rotate(double2 v,int lane){
 if constexpr(S==1)return v;
 int j=lane&(S-1);
 if constexpr(Simple){
  double2 t=__ldg(fft_twiddle+j*(64/(2*S)));double ss=Inverse?-t.y:t.y;
  return make_double2(fma(-ss,v.y,t.x*v.x),fma(ss,v.x,t.x*v.y));
 }

 if(j==0)return v;
 if(j==S/2)return Inverse?make_double2(-v.y,v.x):make_double2(v.y,-v.x);
 if constexpr(S==4){
  constexpr double q=0.707106781186547524400844362104849039;
  double c=j==1?q:-q,s=Inverse?q:-q;
  return make_double2(fma(-s,v.y,c*v.x),fma(s,v.x,c*v.y));
 }else{
  double2 t=__ldg(fft_twiddle+j*(64/(2*S)));double ss=Inverse?-t.y:t.y;
  return make_double2(fma(-ss,v.y,t.x*v.x),fma(ss,v.x,t.x*v.y));
 }
}
template<int N,int Gap> __device__ __forceinline__ double2 f4gap(double2 v,int lane){double2 p=shflz(v,2*Gap,N);if(lane&(2*Gap)){v=subz(p,v);if(lane&Gap)v=make_double2(v.y,-v.x);}else v=addz(v,p);p=shflz(v,Gap,N);return lane&Gap?subz(p,v):addz(v,p);}
template<int N,int Gap> __device__ __forceinline__ double2 i4gap(double2 v,int lane){double2 p=shflz(v,Gap,N);v=lane&Gap?subz(p,v):addz(v,p);p=shflz(v,2*Gap,N);bool hi=lane&(2*Gap);double2 a=hi?p:v,b=hi?v:p;if(lane&Gap)b=make_double2(-b.y,b.x);return hi?subz(a,b):addz(a,b);}
template<int N,bool Inv>__device__ __forceinline__ double2 phaseN(double2 v,int lane){int n1=lane%4,n2=lane/4,k2=__brev((unsigned)n2)>>(N==16?30:29),j=n1*k2;if(j==0)return v;double2 t=__ldg(fft_twiddle+j*(64/N));double si=Inv?-t.y:t.y;return make_double2(fma(-si,v.y,t.x*v.x),fma(si,v.x,t.x*v.y));}
template<int N,int S,bool Simple> __device__ __forceinline__ double2 highforward(double2 v,int lane){double2 p=shflz(v,S*4,N);v=lane&(S*4)?rotate<N/4,S,false,Simple>(subz(p,v),lane/4):addz(v,p);if constexpr(S>1)return highforward<N,S/2,Simple>(v,lane);else return v;}
template<int N,int S,bool Simple> __device__ __forceinline__ double2 highinverse(double2 v,int lane){double2 p=shflz(v,S*4,N);bool hi=lane&(S*4);double2 b=rotate<N/4,S,true,Simple>(hi?v:p,lane/4),a=hi?p:v;v=hi?subz(a,b):addz(a,b);if constexpr(S<N/8)return highinverse<N,S*2,Simple>(v,lane);else return v;}
template<bool Inv,bool Simple> __device__ __forceinline__ double2 phase2(double2 v,int lane){int j=(lane%16)*(lane/16);if constexpr(!Simple){if(j==0)return v;}double2 t=__ldg(fft_twiddle+2*j);double si=Inv?-t.y:t.y;return make_double2(fma(-si,v.y,t.x*v.x),fma(si,v.x,t.x*v.y));}
template<int N,int S,bool Simple=false>
__device__ __forceinline__ double2 forward(double2 v,int lane){
 if constexpr(N==16&&S==8)return f4gap<16,1>(phaseN<16,false>(f4gap<16,4>(v,lane),lane),lane);
 if constexpr(N==32&&S==16){double2 p=shflz(v,16,32);v=lane&16?subz(p,v):addz(v,p);v=phase2<false,Simple>(v,lane);return f4gap<32,1>(phaseN<16,false>(f4gap<32,4>(v,lane),lane),lane);}
 double2 p=shflz(v,S,N);
 if(lane&S)v=rotate<N,S,false,Simple>(subz(p,v),lane);else v=addz(v,p);
 if constexpr(S>1)return forward<N,S/2,Simple>(v,lane);else return v;
}
template<int N,int S,bool Simple=false>
__device__ __forceinline__ double2 inverse(double2 v,int lane){
 if constexpr(N==16&&S==1)return i4gap<16,4>(phaseN<16,true>(i4gap<16,1>(v,lane),lane),lane);
 if constexpr(N==32&&S==1){v=i4gap<32,4>(phaseN<16,true>(i4gap<32,1>(v,lane),lane),lane);v=phase2<true,Simple>(v,lane);double2 p=shflz(v,16,32);return lane&16?subz(p,v):addz(v,p);}
 double2 p=shflz(v,S,N);bool high=lane&S;
 double2 b=rotate<N,S,true,Simple>(high?v:p,lane),a=high?p:v;
 v=high?subz(a,b):addz(a,b);
 if constexpr(S<N/2)return inverse<N,S*2,Simple>(v,lane);else return v;
}


__device__ __forceinline__ int wrap(int x,int w){return x<0?x+w:(x>=w?x-w:x);}
template<int NZ,int NY,int P,int Axis>__device__ __forceinline__ int at(int line,int lane){if constexpr(Axis==0&&P==17)return lane*(NY+1)*17+line;else if constexpr(Axis==1)return ((line/17)*(NY+1)+lane)*P+line%17;else return (lane*(NY+1)+line/17)*P+line%17;}
template<int NZ,int NY,int P,int NW,bool Simple,bool Prune,int FixedR=0,bool Packed=false>
__global__ __launch_bounds__(NW*32,1) void full_fft_kernel(const double* __restrict__ in,double* __restrict__ out,int w,int runtimeR,const double2* __restrict__ h){const int R=FixedR?FixedR:runtimeR;
 constexpr int Size=NZ*(NY+1)*P;extern __shared__ __align__(16) double2 sc[];int tid=threadIdx.x,lane=tid%16,group=tid/16;
 int sx=32-2*R,sy=NY-2*R,sz=NZ-2*R,tx=(w+sx-1)/sx,ty=(w+sy-1)/sy;
 int x0=blockIdx.x%tx*sx,y0=blockIdx.x/tx%ty*sy,z0=blockIdx.x/(tx*ty)*sz;
 static_assert((NW*2)%NY==0);constexpr int INPUT_ZSTEP=NW*2/NY;
 int input_z=wrap(z0+group/NY-R,w),input_y=wrap(y0+group%NY-R,w),input_x=wrap(x0+2*lane-R,w),input_xb=wrap(x0+2*lane+1-R,w);
 long long w2=(long long)w*w,domain=w2*w,input_row=(long long)input_z*w2+(long long)input_y*w;
 int input_p=((group/NY)*(NY+1)+group%NY)*P;
 #pragma unroll 4
 for(int r=0;r<NZ/INPUT_ZSTEP;r++){
  double2 a;if constexpr(Packed)a=reinterpret_cast<const double2*>(in)[(long long)blockIdx.x*NZ*NY*16+(group/NY+r*INPUT_ZSTEP)*NY*16+(group%NY)*16+lane];else a=make_double2(in[input_row+input_x],in[input_row+input_xb]);a=forward<16,8,Simple>(a,lane);int freq=__brev((unsigned)lane)>>28,partner=__brev((unsigned)((16-freq)&15))>>28;
  double2 b=make_double2(__shfl_sync(0xffffffff,a.x,partner,16),-__shfl_sync(0xffffffff,a.y,partner,16)),sum=addz(a,b),diff=subz(a,b),t=__ldg(fft_twiddle+2*freq);
  double2 v=make_double2(.5*(sum.x+t.y*diff.x+t.x*diff.y),.5*(sum.y+t.y*diff.y-t.x*diff.x));sc[input_p+lane]=v;if(lane==0)sc[input_p+16]=make_double2(a.x-a.y,0);
  input_z+=INPUT_ZSTEP;input_row+=INPUT_ZSTEP*w2;if(input_z>=w){input_z-=w;input_row-=domain;}input_p+=INPUT_ZSTEP*(NY+1)*P;
 }
 __syncthreads();
 int ly=tid%NY,gy=tid/NY;constexpr int YG=NW*32/NY;
 int yp=at<NZ,NY,P,1>(gy,ly),xf=gy%17;
 #pragma unroll 4
 for(int line=gy;line<NZ*17;line+=YG){double2 a=forward<NY,NY/2,Simple>(sc[yp],ly);sc[yp]=a;int next=xf+YG%17,carry=next>=17;xf=next-carry*17;yp+=YG/17*(NY+1)*P+YG%17+carry*((NY+1)*P-17);}
 __syncthreads();
 int lz=tid%NZ,gz=tid/NZ;
 #pragma unroll 4
 for(int line=gz;line<NY*17;line+=NW*32/NZ){int p=at<NZ,NY,P,0>(line,lz);double2 a=forward<NZ,NZ/2,Simple>(sc[p],lz),t=__ldg(h+lz*NY*17+line);a=make_double2(fma(-a.y,t.y,a.x*t.x),fma(a.x,t.y,a.y*t.x));a=inverse<NZ,1,Simple>(a,lz);if(!Prune||lz>=2*R){sc[p]=a;}}
 __syncthreads();
 yp=at<NZ,NY,P,1>(gy+(Prune?2*R*17:0),ly);xf=gy%17;
 #pragma unroll 4
 for(int line=gy+(Prune?2*R*17:0);line<NZ*17;line+=YG){double2 a=inverse<NY,1,Simple>(sc[yp],ly);if(!Prune||ly>=2*R)sc[yp]=a;int next=xf+YG%17,carry=next>=17;xf=next-carry*17;yp+=YG/17*(NY+1)*P+YG%17+carry*((NY+1)*P-17);}
 __syncthreads();
 static_assert((NW*2)%NY==0);if(group%NY<2*R)return;
 constexpr int ZSTEP=NW*2/NY;int yy=y0+group%NY-2*R;
 int basep=((2*R+group/NY)*(NY+1)+group%NY)*P;
 long long rowbase=((long long)(z0+group/NY)*w+yy)*w,rowstep=(long long)ZSTEP*w*w;
 #pragma unroll 4
 for(int r=0;r<(NZ-2*R)/ZSTEP;r++){int freq=__brev((unsigned)lane)>>28,p=basep+r*ZSTEP*(NY+1)*P,partner=freq==0?16:(__brev((unsigned)(16-freq))>>28);double2 a=sc[p+lane],b=sc[p+partner];b.y=-b.y;double2 sum=addz(a,b),diff=subz(a,b),t=__ldg(fft_twiddle+2*freq);double2 v=make_double2(sum.x+t.y*diff.x-t.x*diff.y,sum.y+t.y*diff.y+t.x*diff.x);v=inverse<16,1,Simple>(v,lane);
 if constexpr(Packed){if(2*lane>=2*R){long long q=(long long)blockIdx.x*sz*sy*sx+((group/NY+r*ZSTEP)*sy+group%NY-2*R)*sx+2*lane-2*R;reinterpret_cast<double2*>(out)[q/2]=v;}}else{int zz=z0+group/NY+r*ZSTEP,xx=x0+2*lane-2*R;if(zz<w&&yy<w){long long row=rowbase+r*rowstep;if(2*lane>=2*R&&xx<w)out[row+xx]=v.x;if(2*lane+1>=2*R&&xx+1<w)out[row+xx+1]=v.y;}}
 }
}

__global__ void pack_tiles(const double*in,double2*packed,int w,int R){
 int sx=32-2*R,sy=sx,sz=16-2*R,tx=(w+sx-1)/sx,ty=tx;
 int x0=blockIdx.x%tx*sx,y0=blockIdx.x/tx%ty*sy,z0=blockIdx.x/(tx*ty)*sz;
 for(int i=threadIdx.x;i<16*32*16;i+=blockDim.x){int z=i/(32*16),y=i/16%32,x=i%16*2;
  int zz=wrap(z0+z-R,w),yy=wrap(y0+y-R,w),xx=wrap(x0+x-R,w),xb=wrap(x0+x+1-R,w);
  long long row=((long long)zz*w+yy)*w;packed[(long long)blockIdx.x*16*32*16+i]=make_double2(in[row+xx],in[row+xb]);
 }
}
__global__ void unpack_tiles(const double*packed,double*out,int w,int R){
 long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)w*w*w)return;
 int sx=32-2*R,sy=sx,sz=16-2*R,tx=(w+sx-1)/sx,x=p%w,y=p/w%w,z=p/((long long)w*w);
 long long tile=((long long)(z/sz)*tx+y/sy)*tx+x/sx;
 out[p]=packed[tile*sz*sy*sx+((z%sz)*sy+y%sy)*sx+x%sx];
}

} // namespace flashfft3d_detail
struct FlashFFT3DPlan {
 double2* spectrum=nullptr;
 int radius=0,device=-1;
};
static int reverse3d(int x,int bits){int r=0;for(int j=0;j<bits;j++){r=2*r+(x&1);x>>=1;}return r;}
static cudaError_t initialize3d(){
 using namespace flashfft3d_detail;
 static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
 int ready=0;cudaError_t e=cudaMemcpyFromSymbol(&ready,tables_ready,sizeof(ready));if(e!=cudaSuccess||ready)return e;
 double2 tw[64];
 for(int i=0;i<64;i++)tw[i]=make_double2(cos(-2*M_PI*i/64),sin(-2*M_PI*i/64));
 e=cudaMemcpyToSymbol(fft_twiddle,tw,sizeof(tw));if(e!=cudaSuccess)return e;
#define INIT_RADIUS(R) \
 e=cudaFuncSetAttribute(full_fft_kernel<16,32,17,32,false,true,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,2*16*33*17*sizeof(double));if(e!=cudaSuccess)return e; \
 e=cudaFuncSetAttribute(full_fft_kernel<16,32,17,32,false,true,R,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,2*16*33*17*sizeof(double));if(e!=cudaSuccess)return e;
 INIT_RADIUS(1) INIT_RADIUS(2) INIT_RADIUS(3)
#undef INIT_RADIUS
 ready=1;return cudaMemcpyToSymbol(tables_ready,&ready,sizeof(ready));
}
cudaError_t flashfft3dCreatePlan(FlashFFT3DPlan** result,const double*coeff,int radius,FlashFFT3DBackend backend){
 if(!result)return cudaErrorInvalidValue;*result=nullptr;
 if(!coeff||radius<1||radius>3)return cudaErrorInvalidValue;
 // Preserve the legacy selector, but never silently substitute another algorithm.
 if(backend==FlashFFT3DBackend::XYFFT)return cudaErrorNotSupported;
 if(backend!=FlashFFT3DBackend::FullFFT)return cudaErrorInvalidValue;
 int k=2*radius+1;for(int i=0;i<k*k*k;i++)if(!std::isfinite(coeff[i]))return cudaErrorInvalidValue;
 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 cudaDeviceProp prop;e=cudaGetDeviceProperties(&prop,device);if(e!=cudaSuccess)return e;
 if(prop.major!=8||prop.minor!=0)return cudaErrorNotSupported;
 e=initialize3d();if(e!=cudaSuccess)return e;
 auto*plan=new(std::nothrow) FlashFFT3DPlan;if(!plan)return cudaErrorMemoryAllocation;
 plan->radius=radius;plan->device=device;
 std::vector<double2>h(16*32*17);
  for(int z=0;z<16;z++)for(int y=0;y<32;y++)for(int x=0;x<17;x++){
   int kz=reverse3d(z,4),ky=reverse3d(y,5),kx=x<16?reverse3d(x,4):16;double re=0,im=0;
   for(int a=0;a<k;a++)for(int b=0;b<k;b++)for(int d=0;d<k;d++){
    double angle=-2*M_PI*((double)kz*a/16+(double)ky*b/32+(double)kx*d/32),v=coeff[((k-1-a)*k+k-1-b)*k+k-1-d];re+=v*cos(angle);im+=v*sin(angle);
   }h[(z*32+y)*17+x]=make_double2(re/16384,im/16384);
  }
 for(auto v:h)if(!std::isfinite(v.x)||!std::isfinite(v.y)){delete plan;return cudaErrorInvalidValue;}
 e=cudaMalloc(&plan->spectrum,h.size()*sizeof(double2));if(e!=cudaSuccess){delete plan;return e;}
 e=cudaMemcpy(plan->spectrum,h.data(),h.size()*sizeof(double2),cudaMemcpyHostToDevice);if(e!=cudaSuccess){cudaFree(plan->spectrum);delete plan;return e;}
 *result=plan;return cudaSuccess;
}
cudaError_t flashfft3dExecute(const FlashFFT3DPlan*plan,const double*input,double*output,int width,cudaStream_t stream){
 using namespace flashfft3d_detail;
 if(!plan||!input||!output||input==output||width<64)return cudaErrorInvalidValue;
 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=plan->device)return cudaErrorInvalidDevice;
 const long long sub=32-2*plan->radius,tiles=((long long)width+sub-1)/sub;
 const long long maximum=std::numeric_limits<int>::max();
 if(tiles>maximum/tiles)return cudaErrorInvalidValue;
 long long planeCount=tiles*tiles;
 long long slices=((long long)width+15-2*plan->radius)/(16-2*plan->radius);
 if(slices>maximum/planeCount)return cudaErrorInvalidValue;
 long long count=planeCount*slices;
#define EXEC_RADIUS(R) \
 case R:full_fft_kernel<16,32,17,32,false,true,R><<<static_cast<unsigned>(count),1024,2*16*33*17*sizeof(double),stream>>>(input,output,width,R,plan->spectrum);break;
 switch(plan->radius){EXEC_RADIUS(1) EXEC_RADIUS(2) EXEC_RADIUS(3) default:return cudaErrorInvalidValue;}
#undef EXEC_RADIUS
 return cudaGetLastError();
}
cudaError_t flashfft3dDestroyPlan(FlashFFT3DPlan*plan){
 if(!plan)return cudaSuccess;int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=plan->device)return cudaErrorInvalidDevice;
 e=cudaFree(plan->spectrum);if(e==cudaSuccess)delete plan;return e;
}

struct FlashFFT3DWorkspace {double*input=nullptr;double*output=nullptr;int width=0,radius=0,device=-1;unsigned count=0;};
static cudaError_t validateSplit(const FlashFFT3DPlan*plan,const FlashFFT3DWorkspace*workspace){
 if(!plan||!workspace||workspace->radius!=plan->radius)return cudaErrorInvalidValue;
 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 return device==plan->device&&device==workspace->device?cudaSuccess:cudaErrorInvalidDevice;
}
cudaError_t flashfft3dCreateWorkspace(FlashFFT3DWorkspace**result,const FlashFFT3DPlan*plan,int width){
 if(!result)return cudaErrorInvalidValue;*result=nullptr;if(!plan||width<64)return cudaErrorInvalidValue;
 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=plan->device)return cudaErrorInvalidDevice;
 long long sub=32-2*plan->radius,tiles=((long long)width+sub-1)/sub,count,inputElements,outputElements;
 if(tiles>2147483647/tiles)return cudaErrorInvalidValue;
 long long slices=((long long)width+15-2*plan->radius)/(16-2*plan->radius);if(tiles*tiles>2147483647/slices)return cudaErrorInvalidValue;count=tiles*tiles*slices;inputElements=count*16384;outputElements=count*(16-2*plan->radius)*sub*sub;
 if(count>2147483647||((long long)width*width*width+255)/256>2147483647)return cudaErrorInvalidValue;
 auto*ws=new(std::nothrow) FlashFFT3DWorkspace;if(!ws)return cudaErrorMemoryAllocation;
 ws->width=width;ws->radius=plan->radius;ws->device=device;ws->count=static_cast<unsigned>(count);
 e=cudaMalloc(&ws->input,static_cast<size_t>(inputElements)*sizeof(double));if(e!=cudaSuccess){delete ws;return e;}
 e=cudaMalloc(&ws->output,static_cast<size_t>(outputElements)*sizeof(double));if(e!=cudaSuccess){cudaFree(ws->input);delete ws;return e;}
 *result=ws;return cudaSuccess;
}
cudaError_t flashfft3dPack(const FlashFFT3DPlan*plan,FlashFFT3DWorkspace*workspace,const double*input,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;if(!input)return cudaErrorInvalidValue;
 flashfft3d_detail::pack_tiles<<<workspace->count,256,0,stream>>>(input,reinterpret_cast<double2*>(workspace->input),workspace->width,plan->radius);
 return cudaGetLastError();
}
cudaError_t flashfft3dCompute(const FlashFFT3DPlan*plan,FlashFFT3DWorkspace*workspace,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;
 using namespace flashfft3d_detail;
#define SPLIT_RADIUS(R) case R:full_fft_kernel<16,32,17,32,false,true,R,true><<<workspace->count,1024,2*16*33*17*sizeof(double),stream>>>(workspace->input,workspace->output,workspace->width,R,plan->spectrum);break;
 switch(plan->radius){SPLIT_RADIUS(1) SPLIT_RADIUS(2) SPLIT_RADIUS(3) default:return cudaErrorInvalidValue;}
#undef SPLIT_RADIUS
 return cudaGetLastError();
}
cudaError_t flashfft3dUnpack(const FlashFFT3DPlan*plan,const FlashFFT3DWorkspace*workspace,double*output,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;if(!output)return cudaErrorInvalidValue;
 long long count=(long long)workspace->width*workspace->width*workspace->width;
 flashfft3d_detail::unpack_tiles<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(workspace->output,output,workspace->width,plan->radius);
 return cudaGetLastError();
}
cudaError_t flashfft3dDestroyWorkspace(FlashFFT3DWorkspace*workspace){
 if(!workspace)return cudaSuccess;int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=workspace->device)return cudaErrorInvalidDevice;
 e=cudaFree(workspace->input);cudaError_t second=cudaFree(workspace->output);delete workspace;return e!=cudaSuccess?e:second;
}
