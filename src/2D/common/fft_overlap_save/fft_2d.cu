#include "fft_2d.cuh"
#include <vector>
#include <new>
#include <mutex>
#include <cmath>
namespace flashfft2d_detail {
__device__ double2 fft_twiddle[64];
__device__ int tables_ready=0;
__device__ __align__(32) double tc_global_re[64],tc_global_im[64];
static __device__ __forceinline__ int periodic(int x,int w){if(x<0)x+=w;else if(x>=w)x-=w;return x;}

__global__ void pack_tiles(const double*in,double2*packed,int w,int R){
 int sub=32-2*R,tiles=(w+sub-1)/sub,pairs=(tiles+1)/2;
 int y0=blockIdx.x/pairs*sub,x0=blockIdx.x%pairs*2*sub;
 for(int i=threadIdx.x;i<1024;i+=blockDim.x){int row=i/32,col=i%32;
  int y=periodic(y0+row-R,w),x=periodic(x0+col-R,w),xx=periodic(x0+sub+col-R,w);
  packed[(long long)blockIdx.x*1024+i]=make_double2(in[(long long)y*w+x],in[(long long)y*w+xx]);
 }
}
__global__ void unpack_tiles(const double2*packed,double*out,int w,int R){
 long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)w*w)return;
 int sub=32-2*R,tiles=(w+sub-1)/sub,pairs=(tiles+1)/2,y=p/w,x=p%w,tx=x/sub,ty=y/sub;
 double2 v=packed[((long long)ty*pairs+tx/2)*1024+(y%sub+2*R)*32+x%sub+2*R];out[p]=(tx&1)?v.y:v.x;
}
#include "register_fft.cuh"
} // namespace flashfft2d_detail
struct FlashFFT2DPlan {double2* spectrum=nullptr;int radius=0;int device=-1;};
cudaError_t flashfft2dCreatePlan(FlashFFT2DPlan** result,const double* coeff,int radius){
 if(!result)return cudaErrorInvalidValue;*result=nullptr;
 if(!coeff||radius<1||radius>3)return cudaErrorInvalidValue;
 int k=2*radius+1;for(int i=0;i<k*k;i++)if(!std::isfinite(coeff[i]))return cudaErrorInvalidValue;
 int device;cudaError_t err=cudaGetDevice(&device);if(err!=cudaSuccess)return err;
 cudaDeviceProp prop;err=cudaGetDeviceProperties(&prop,device);if(err!=cudaSuccess)return err;
 // This backend uses FP64 Tensor Core MMA, tuned and verified on A100 (sm_80).
 if(prop.major!=8||prop.minor!=0)return cudaErrorNotSupported;
 static std::mutex init_mutex;
 {std::lock_guard<std::mutex> lock(init_mutex);int ready=0;err=cudaMemcpyFromSymbol(&ready,flashfft2d_detail::tables_ready,sizeof(ready));if(err!=cudaSuccess)return err;if(!ready){
  double re[64],im[64];double2 tw[64];
  for(int i=0;i<8;i++)for(int j=0;j<8;j++){re[i*8+j]=cos(-2*M_PI*i*j/8);im[i*8+j]=sin(-2*M_PI*i*j/8);}
  for(int i=0;i<64;i++)tw[i]=make_double2(cos(-2*M_PI*i/64),sin(-2*M_PI*i/64));
  err=cudaMemcpyToSymbol(flashfft2d_detail::tc_global_re,re,sizeof(re));if(err!=cudaSuccess)return err;
  err=cudaMemcpyToSymbol(flashfft2d_detail::tc_global_im,im,sizeof(im));if(err!=cudaSuccess)return err;
  err=cudaMemcpyToSymbol(flashfft2d_detail::fft_twiddle,tw,sizeof(tw));if(err!=cudaSuccess)return err;
  ready=1;err=cudaMemcpyToSymbol(flashfft2d_detail::tables_ready,&ready,sizeof(ready));if(err!=cudaSuccess)return err;
 }}
 auto* plan=new(std::nothrow) FlashFFT2DPlan;if(!plan)return cudaErrorMemoryAllocation;plan->radius=radius;plan->device=device;
 std::vector<double2> h(32*32);
 for(int col=0;col<32;col++)for(int row=0;row<32;row++){
  int ky=row,kx=col;double re=0,im=0;
  for(int y=0;y<k;y++)for(int x=0;x<k;x++){double a=-2*M_PI*(ky*y+kx*x)/32,v=coeff[(k-1-y)*k+k-1-x];re+=v*cos(a);im+=v*sin(a);}
  h[col*32+row]=make_double2(re/1024,im/1024);
 }
 err=cudaMalloc(&plan->spectrum,h.size()*sizeof(double2));if(err!=cudaSuccess){delete plan;return err;}
 err=cudaMemcpy(plan->spectrum,h.data(),h.size()*sizeof(double2),cudaMemcpyHostToDevice);if(err!=cudaSuccess){cudaFree(plan->spectrum);delete plan;return err;}
 *result=plan;return cudaSuccess;
}
cudaError_t flashfft2dExecute(const FlashFFT2DPlan* plan,const double* input,double* output,int width,cudaStream_t stream){
 if(!plan||!input||!output||input==output||width<64)return cudaErrorInvalidValue;
 int device;cudaError_t err=cudaGetDevice(&device);if(err!=cudaSuccess)return err;if(device!=plan->device)return cudaErrorInvalidDevice;
 int sub=32-2*plan->radius;long long tiles=((long long)width+sub-1)/sub,count=tiles*((tiles+1)/2);if(count>2147483647)return cudaErrorInvalidValue;
 if(width&1)flashfft2d_detail::register_fft::kernel<false,false,35,1><<<static_cast<unsigned>(count),128,32*35*sizeof(double2),stream>>>(input,output,width,plan->radius,plan->spectrum);
 else flashfft2d_detail::register_fft::kernel<false,true,35,1><<<static_cast<unsigned>(count),128,32*35*sizeof(double2),stream>>>(input,output,width,plan->radius,plan->spectrum);
 return cudaGetLastError();
}
cudaError_t flashfft2dDestroyPlan(FlashFFT2DPlan* plan){
 if(!plan)return cudaSuccess;int device;cudaError_t err=cudaGetDevice(&device);if(err!=cudaSuccess)return err;if(device!=plan->device)return cudaErrorInvalidDevice;
 err=cudaFree(plan->spectrum);if(err==cudaSuccess)delete plan;return err;
}

struct FlashFFT2DWorkspace {double*input=nullptr;double*output=nullptr;int width=0,radius=0,device=-1;unsigned count=0;};
static cudaError_t validateSplit(const FlashFFT2DPlan*plan,const FlashFFT2DWorkspace*workspace){
 if(!plan||!workspace||workspace->radius!=plan->radius)return cudaErrorInvalidValue;

 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 return device==plan->device&&device==workspace->device?cudaSuccess:cudaErrorInvalidDevice;
}
cudaError_t flashfft2dCreateWorkspace(FlashFFT2DWorkspace**result,const FlashFFT2DPlan*plan,int width){
 if(!result)return cudaErrorInvalidValue;*result=nullptr;if(!plan||width<64)return cudaErrorInvalidValue;

 int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=plan->device)return cudaErrorInvalidDevice;
 long long sub=32-2*plan->radius,tiles=((long long)width+sub-1)/sub,count,inputElements,outputElements;
 if(tiles>2147483647/tiles)return cudaErrorInvalidValue;
 count=tiles*((tiles+1)/2);inputElements=count*2048;outputElements=count*2048;
 if(count>2147483647||((long long)width*width+255)/256>2147483647)return cudaErrorInvalidValue;
 auto*ws=new(std::nothrow) FlashFFT2DWorkspace;if(!ws)return cudaErrorMemoryAllocation;
 ws->width=width;ws->radius=plan->radius;ws->device=device;ws->count=static_cast<unsigned>(count);
 e=cudaMalloc(&ws->input,static_cast<size_t>(inputElements)*sizeof(double));if(e!=cudaSuccess){delete ws;return e;}
 e=cudaMalloc(&ws->output,static_cast<size_t>(outputElements)*sizeof(double));if(e!=cudaSuccess){cudaFree(ws->input);delete ws;return e;}
 *result=ws;return cudaSuccess;
}
cudaError_t flashfft2dPack(const FlashFFT2DPlan*plan,FlashFFT2DWorkspace*workspace,const double*input,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;if(!input)return cudaErrorInvalidValue;
 flashfft2d_detail::pack_tiles<<<workspace->count,256,0,stream>>>(input,reinterpret_cast<double2*>(workspace->input),workspace->width,plan->radius);
 return cudaGetLastError();
}
cudaError_t flashfft2dCompute(const FlashFFT2DPlan*plan,FlashFFT2DWorkspace*workspace,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;
 flashfft2d_detail::register_fft::kernel<false,false,35,1,true><<<workspace->count,128,32*35*sizeof(double2),stream>>>(workspace->input,workspace->output,workspace->width,plan->radius,plan->spectrum);
 return cudaGetLastError();
}
cudaError_t flashfft2dUnpack(const FlashFFT2DPlan*plan,const FlashFFT2DWorkspace*workspace,double*output,cudaStream_t stream){
 cudaError_t e=validateSplit(plan,workspace);if(e!=cudaSuccess)return e;if(!output)return cudaErrorInvalidValue;
 long long count=(long long)workspace->width*workspace->width;
 flashfft2d_detail::unpack_tiles<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(reinterpret_cast<const double2*>(workspace->output),output,workspace->width,plan->radius);
 return cudaGetLastError();
}
cudaError_t flashfft2dDestroyWorkspace(FlashFFT2DWorkspace*workspace){
 if(!workspace)return cudaSuccess;int device;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;if(device!=workspace->device)return cudaErrorInvalidDevice;
 e=cudaFree(workspace->input);cudaError_t second=cudaFree(workspace->output);delete workspace;return e!=cudaSuccess?e:second;
}
