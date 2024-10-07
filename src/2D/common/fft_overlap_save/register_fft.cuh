// FP64 register-held Tensor Core FFT for the fused overlap-save path.
// Included inside flashfft2d_detail after its immutable DFT tables.
namespace register_fft {
// SM80 FP64 MMA: lane/4 selects the A row; lane%4 selects K.
// Each lane owns output columns 2*(lane%4) and 2*(lane%4)+1.
__device__ __forceinline__ void mma(double a,double b,double&c0,double&c1){
 asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};":"=d"(c0),"=d"(c1):"d"(a),"d"(b),"d"(c0),"d"(c1));
}
__device__ __forceinline__ double2 add(double2 a,double2 b){return make_double2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ double2 sub(double2 a,double2 b){return make_double2(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ double2 phase(double2 v,int j,bool inv){double2 t=__ldg(fft_twiddle+2*j);double s=inv?-t.y:t.y;return make_double2(fma(-s,v.y,t.x*v.x),fma(s,v.x,t.x*v.y));}
__device__ __forceinline__ int wrap(int x,int w){return x<0?x+w:(x>=w?x-w:x);}
// Permute column bits, then XOR with row bits; every row is a bijection.
__device__ __forceinline__ int sy(int y,int row){return ((4*(y%4)+(y/4)%4+16*(y/16))^((row/2)&7)^(4*(row%4)))&31;}
template<bool Inverse>__device__ __forceinline__ void fft4(double2(&v)[4][2],int q,int k2){
 if constexpr(!Inverse){
  #pragma unroll
  for(int n=1;n<4;n++)v[n][q]=phase(v[n][q],n*k2,false);
  double2 a=add(v[0][q],v[2][q]),b=sub(v[0][q],v[2][q]),c=add(v[1][q],v[3][q]),d=sub(v[1][q],v[3][q]);d=make_double2(d.y,-d.x);
  v[0][q]=add(a,c);v[2][q]=sub(a,c);v[1][q]=add(b,d);v[3][q]=sub(b,d);
 }else{
  double2 a=add(v[0][q],v[2][q]),b=sub(v[0][q],v[2][q]),c=add(v[1][q],v[3][q]),d=sub(v[1][q],v[3][q]);d=make_double2(-d.y,d.x);
  v[0][q]=add(a,c);v[2][q]=sub(a,c);v[1][q]=add(b,d);v[3][q]=sub(b,d);
  #pragma unroll
  for(int n=1;n<4;n++)v[n][q]=phase(v[n][q],n*k2,true);
 }
}
template<bool Gauss>__device__ __forceinline__ void accum(double ar,double ai,double br,double bi,double(&cr)[2],double(&ci)[2],double(&cq)[2]){
 if constexpr(Gauss){mma(ar,br,cr[0],cr[1]);mma(ai,bi,cq[0],cq[1]);mma(ar+ai,br+bi,ci[0],ci[1]);}
 else{mma(ar,br,cr[0],cr[1]);mma(ar,bi,ci[0],ci[1]);mma(ai,br,ci[0],ci[1]);mma(-ai,bi,cr[0],cr[1]);}
}
template<bool Gauss>__device__ __forceinline__ void finish(double(&cr)[2],double(&ci)[2],double(&cq)[2]){if constexpr(Gauss){
 #pragma unroll
 for(int q=0;q<2;q++){ci[q]=(ci[q]-cr[q])-cq[q];cr[q]-=cq[q];}
}}
template<bool Gauss,bool Even,int Pad=33,int MB=1,bool Packed=false>__global__ __launch_bounds__(128,MB) void kernel(const double*in,double*out,int w,int R,const double2*h){
 extern __shared__ __align__(16) double2 sc[];int lane=threadIdx.x%32,warp=threadIdx.x/32,m=lane/4,t=lane%4,row=warp*8+m;
 int subw=32-2*R,tiles=(w+subw-1)/subw,pairs=(tiles+1)/2,x0=blockIdx.x%pairs*2*subw,y0=blockIdx.x/pairs*subw;
 double br[2],bi[2];
 #pragma unroll
 for(int p=0;p<2;p++){br[p]=__ldg(tc_global_re+(t+4*p)*8+m);bi[p]=__ldg(tc_global_im+(t+4*p)*8+m);}
 // Forward X: load A fragments directly from the two real input tiles.
 double2 v[4][2];
 #pragma unroll
 for(int n1=0;n1<4;n1++){
  double cr[2]={},ci[2]={},cq[2]={};
  #pragma unroll
  for(int p=0;p<2;p++){
   int yy=wrap(y0+row-R,w),col=4*(t+4*p)+n1,xx=wrap(x0+col-R,w),xb=wrap(x0+subw+col-R,w);long long base=(long long)yy*w;
   if constexpr(Packed){double2 a=reinterpret_cast<const double2*>(in)[(long long)blockIdx.x*1024+row*32+col];accum<Gauss>(a.x,a.y,br[p],bi[p],cr,ci,cq);}
   else accum<Gauss>(in[base+xx],in[base+xb],br[p],bi[p],cr,ci,cq);
  }finish<Gauss>(cr,ci,cq);
  #pragma unroll
  for(int q=0;q<2;q++)v[n1][q]=make_double2(cr[q],ci[q]);
 }
 #pragma unroll
 for(int q=0;q<2;q++)fft4<false>(v,q,2*t+q);
 #pragma unroll
 for(int n=0;n<4;n++){
  #pragma unroll
  for(int q=0;q<2;q++){int fx=2*t+q+8*n;sc[fx*Pad+sy(row,fx)]=v[n][q];}
 }
 __syncthreads();
 // Forward Y, spectrum multiplication, and inverse 4-point butterflies.
 #pragma unroll
 for(int n1=0;n1<4;n1++){
  double cr[2]={},ci[2]={},cq[2]={};
  #pragma unroll
  for(int p=0;p<2;p++){double2 a=sc[row*Pad+sy(4*(t+4*p)+n1,row)];accum<Gauss>(a.x,a.y,br[p],bi[p],cr,ci,cq);}
  finish<Gauss>(cr,ci,cq);
  #pragma unroll
  for(int q=0;q<2;q++)v[n1][q]=make_double2(cr[q],ci[q]);
 }
 #pragma unroll
 for(int q=0;q<2;q++)fft4<false>(v,q,2*t+q);
 #pragma unroll
 for(int n=0;n<4;n++){
  #pragma unroll
  for(int q=0;q<2;q++){double2 a=v[n][q],b=__ldg(h+row*32+2*t+q+8*n);v[n][q]=make_double2(fma(-a.y,b.y,a.x*b.x),fma(a.x,b.y,a.y*b.x));}
 }
 #pragma unroll
 for(int q=0;q<2;q++)fft4<true>(v,q,2*t+q);
 __syncwarp();
 #pragma unroll
 for(int n=0;n<4;n++){
  #pragma unroll
  for(int q=0;q<2;q++)sc[row*Pad+sy(4*(2*t+q)+n,row)]=v[n][q];
 }
 __syncwarp();
 #pragma unroll
 for(int n1=0;n1<4;n1++){
  double cr[2]={},ci[2]={},cq[2]={};
  #pragma unroll
  for(int p=0;p<2;p++){double2 a=sc[row*Pad+sy(4*(t+4*p)+n1,row)];accum<Gauss>(a.x,a.y,br[p],-bi[p],cr,ci,cq);}
  finish<Gauss>(cr,ci,cq);__syncwarp();
  #pragma unroll
  for(int q=0;q<2;q++)sc[row*Pad+sy(4*(2*t+q)+n1,row)]=make_double2(cr[q],ci[q]);
  __syncwarp();
 }
 __syncthreads();
 #pragma unroll
 for(int n=0;n<4;n++){
  #pragma unroll
  for(int q=0;q<2;q++){int fx=2*t+q+8*n;v[n][q]=sc[fx*Pad+sy(row,fx)];}
 }
 #pragma unroll
 for(int q=0;q<2;q++)fft4<true>(v,q,2*t+q);
 __syncthreads();
 #pragma unroll
 for(int n=0;n<4;n++){
  #pragma unroll
  for(int q=0;q<2;q++)sc[row*Pad+sy(4*(2*t+q)+n,row)]=v[n][q];
 }
 __syncwarp();
 #pragma unroll
 for(int n1=0;n1<4;n1++){
  double cr[2]={},ci[2]={},cq[2]={};
  #pragma unroll
  for(int p=0;p<2;p++){double2 a=sc[row*Pad+sy(4*(t+4*p)+n1,row)];accum<Gauss>(a.x,a.y,br[p],-bi[p],cr,ci,cq);}
  finish<Gauss>(cr,ci,cq);
  #pragma unroll
  for(int q=0;q<2;q++)v[n1][q]=make_double2(cr[q],ci[q]);
 }
 if constexpr(Packed){
  if(row>=2*R){
   #pragma unroll
   for(int q=0;q<2;q++){
    #pragma unroll
    for(int n=0;n<4;n++){int col=4*(2*t+q)+n;if(col>=2*R)reinterpret_cast<double2*>(out)[(long long)blockIdx.x*1024+row*32+col]=v[n][q];}
   }
  }
 }else if(row>=2*R&&y0+row-2*R<w){int yy=y0+row-2*R;long long base=(long long)yy*w;
  if constexpr(!Even){
   #pragma unroll
   for(int q=0;q<2;q++){
    #pragma unroll
    for(int n=0;n<4;n++){int col=4*(2*t+q)+n,xx=x0+col-2*R,xb=xx+subw;if(col>=2*R){if(xx<w)out[base+xx]=v[n][q].x;if(xb<w)out[base+xb]=v[n][q].y;}}
   }
  }else{
   #pragma unroll
   for(int q=0;q<2;q++){
    #pragma unroll
    for(int n=0;n<4;n+=2){int col=4*(2*t+q)+n,xx=x0+col-2*R,xb=xx+subw;if(col>=2*R){if(xx+1<w)reinterpret_cast<double2*>(out+base+xx)[0]=make_double2(v[n][q].x,v[n+1][q].x);if(xb+1<w)reinterpret_cast<double2*>(out+base+xb)[0]=make_double2(v[n][q].y,v[n+1][q].y);}}
   }
  }
 }
}
} // namespace register_fft
