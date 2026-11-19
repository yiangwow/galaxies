// galaxies: self-gravitating galaxy collisions, live on the GPU.
// Gravity: particle-mesh. Stars are deposited on a 256^3 grid, the Poisson equation is solved
// with a multigrid V-cycle in compute shaders, and every star reads its pull back from the grid.
// Build: make.   Run: ./galaxies   (click = throw a galaxy in, mouse = look, wheel = zoom)
#include <epoxy/gl.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <vector>

static const int N = 256, LEVELS = 7;   // grid 256^3, coarsest level 4^3
static const int SLOTS = 4;             // galaxies alive at once
static int NSTAR = 1 << 20, NHALO = 1 << 18, NGAL, NTOT;
static const float PI = 3.14159265f, GRAV = 1.0f, SPEED = 30.0f;   // SPEED: sim time units per second
static const float SPAWN_R = 75.0f;

// ---------------------------------------------------------------- shaders
static const char* CS_UPDATE = R"(
layout(local_size_x=256) in;
layout(std430,binding=0) buffer P { vec4 pos[]; };
layout(std430,binding=1) buffer V { vec4 vel[]; };
layout(binding=0) uniform sampler3D uForce;
layout(binding=0,r32ui) uniform uimage3D uDens;
uniform float uDt, uUnits; uniform vec3 uShift, uKick; uniform uint uCount;
void main(){
  uint i=gl_GlobalInvocationID.x; if(i>=uCount) return;
  vec4 p=pos[i]; if(p.w<=0.0) return;
  vec4 v=vel[i];
  vec3 a=textureLod(uForce,p.xyz/256.0,0.0).xyz;
  v.xyz+=a*uDt+uKick; p.xyz+=v.xyz*uDt+uShift;
  pos[i]=p; vel[i]=v;
  vec3 g=p.xyz-0.5, fl=floor(g), f=g-fl; ivec3 b=ivec3(fl); float u=p.w*uUnits;
  for(int k=0;k<8;k++){
    ivec3 o=ivec3(k&1,(k>>1)&1,(k>>2)&1), c=b+o;
    if(any(lessThan(c,ivec3(0)))||any(greaterThanEqual(c,ivec3(256)))) continue;
    vec3 w3=mix(1.0-f,f,vec3(o)); uint w=uint(w3.x*w3.y*w3.z*u+0.5);
    if(w>0u) imageAtomicAdd(uDens,c,w);
  }
}
)";

static const char* CS_SPAWN = R"(
layout(local_size_x=256) in;
layout(std430,binding=0) buffer P { vec4 pos[]; };
layout(std430,binding=1) buffer V { vec4 vel[]; };
layout(std430,binding=2) readonly buffer TP { vec4 tpos[]; };
layout(std430,binding=3) readonly buffer TV { vec4 tvel[]; };
layout(std430,binding=4) buffer C { uint colr[]; };
layout(std430,binding=5) readonly buffer TC { uint tcol[]; };
uniform uint uOffset, uCount, uTpl; uniform mat3 uRot; uniform vec3 uPos, uVel;
void main(){
  uint i=gl_GlobalInvocationID.x; if(i>=uCount) return;
  vec4 p=tpos[uTpl+i], v=tvel[uTpl+i];
  pos[uOffset+i]=vec4(uRot*p.xyz+uPos,p.w);
  vel[uOffset+i]=vec4(uRot*v.xyz+uVel,0.0);
  colr[uOffset+i]=tcol[uTpl+i];
}
)";

static const char* CS_RHS = R"(
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(binding=0,r32ui) readonly uniform uimage3D uDens;
layout(binding=1,r32f) writeonly uniform image3D uRhs;
uniform float uScale;
void main(){ ivec3 c=ivec3(gl_GlobalInvocationID); imageStore(uRhs,c,vec4(float(imageLoad(uDens,c).r)*uScale)); }
)";

// Shared: neighbour read with a zero-potential wall on the box faces.
#define GLSL_NB R"(
float nb(ivec3 p,float c){ return (any(lessThan(p,ivec3(0)))||any(greaterThanEqual(p,ivec3(uSize)))) ? -c : imageLoad(uPhi,p).r; }
float sum6(ivec3 c,float v){ return nb(c+ivec3(1,0,0),v)+nb(c-ivec3(1,0,0),v)+nb(c+ivec3(0,1,0),v)+nb(c-ivec3(0,1,0),v)+nb(c+ivec3(0,0,1),v)+nb(c-ivec3(0,0,1),v); }
)"

static const char* CS_SMOOTH = R"(
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(binding=0,r32f) readonly uniform image3D uPhi;
layout(binding=1,r32f) readonly uniform image3D uRhs;
layout(binding=2,r32f) writeonly uniform image3D uDst;
uniform int uSize; uniform float uH2;
)" GLSL_NB R"(
void main(){
  ivec3 c=ivec3(gl_GlobalInvocationID); if(any(greaterThanEqual(c,ivec3(uSize)))) return;
  float v=imageLoad(uPhi,c).r;
  float j=(sum6(c,v)-uH2*imageLoad(uRhs,c).r)/6.0;
  imageStore(uDst,c,vec4(mix(v,j,0.857)));
}
)";

static const char* CS_RESTRICT = R"(
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(binding=0,r32f) readonly uniform image3D uPhi;
layout(binding=1,r32f) readonly uniform image3D uRhs;
layout(binding=2,r32f) writeonly uniform image3D uOut;
uniform int uSize; uniform float uInvH2;
)" GLSL_NB R"(
void main(){
  ivec3 C=ivec3(gl_GlobalInvocationID); if(any(greaterThanEqual(C,ivec3(uSize/2)))) return;
  float acc=0.0;
  for(int k=0;k<8;k++){
    ivec3 f=2*C+ivec3(k&1,(k>>1)&1,(k>>2)&1); float v=imageLoad(uPhi,f).r;
    acc+=imageLoad(uRhs,f).r-(sum6(f,v)-6.0*v)*uInvH2;
  }
  imageStore(uOut,C,vec4(acc*0.125));
}
)";

static const char* CS_PROLONG = R"(
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(binding=0) uniform sampler3D uCoarse;
layout(binding=0,r32f) uniform image3D uPhi;
uniform int uSize;
void main(){
  ivec3 c=ivec3(gl_GlobalInvocationID); if(any(greaterThanEqual(c,ivec3(uSize)))) return;
  float e=textureLod(uCoarse,(vec3(c)+0.5)/float(uSize),0.0).r;
  imageStore(uPhi,c,vec4(imageLoad(uPhi,c).r+e));
}
)";

static const char* CS_GRAD = R"(
layout(local_size_x=8,local_size_y=8,local_size_z=8) in;
layout(binding=0,r32f) readonly uniform image3D uPhi;
layout(binding=1,rgba16f) writeonly uniform image3D uForceOut;
uniform int uSize;
)" GLSL_NB R"(
void main(){
  ivec3 c=ivec3(gl_GlobalInvocationID); float v=imageLoad(uPhi,c).r;
  vec3 g=vec3(nb(c+ivec3(1,0,0),v)-nb(c-ivec3(1,0,0),v), nb(c+ivec3(0,1,0),v)-nb(c-ivec3(0,1,0),v), nb(c+ivec3(0,0,1),v)-nb(c-ivec3(0,0,1),v));
  imageStore(uForceOut,c,vec4(-0.5*g,0.0));
}
)";

static const char* VS_STARS = R"(
layout(std430,binding=0) readonly buffer P { vec4 pos[]; };
layout(std430,binding=4) readonly buffer C { uint colr[]; };
layout(binding=0) uniform sampler3D uRho;
uniform mat4 uVP; uniform float uFlux, uSize, uRhoRef, uTime, uNova;
out vec3 vCol; out float vNova;
float hash(uint x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return float(x)/4294967296.0; }
void main(){
  uint id=uint(gl_VertexID); vec4 p=pos[id]; uint c=colr[id];
  if(p.w<=0.0||(c>>31)==1u){ gl_Position=vec4(2.0,2.0,2.0,1.0); gl_PointSize=1.0; vCol=vec3(0.0); return; }
  gl_Position=uVP*vec4(p.xyz-128.0,1.0);
  float d=gl_Position.w;
  vec3 col=vec3(float(c&255u),float((c>>8)&255u),float((c>>16)&255u))/127.0;
  float sens=float((c>>28)&3u)*0.5; bool knot=((c>>30)&1u)==1u;
  float h=hash(id), h2=hash(id*7919u+13u);
  // crowded gas makes young blue stars: brighten and blue-shift gas-rich stars where the grid is dense
  float rho=textureLod(uRho,p.xyz/256.0,0.0).r/uRhoRef;
  float young=min(1.0,sens*smoothstep(0.6,3.0,rho));
  col=mix(col,vec3(0.50,0.68,1.0)*max(col.r,max(col.g,col.b)),young*0.6);
  float lum=(0.45+0.9*h2)*(1.0+2.2*young), big=1.0;
  if(knot){ lum*=5.0; big=1.7; } else if(h>0.992){ lum*=7.0; big=1.7; }
  // supernovae: every star owns one moment in a 90 second cycle; few take it, far more where young stars crowd
  // gas-rich stars at disc-and-arm densities; the packed cores are left out (a flash there is white on white)
  float burst=min(1.0,sens)*smoothstep(0.03,0.4,rho)*(1.0-smoothstep(1.5,6.0,rho));
  float chance=(0.000012+0.0015*burst)*uNova; vNova=0.0; float flash=0.0;
  if(hash(id*131u+71u)<chance){
    float age=fract(uTime/90.0+hash(id*31u+7u))*90.0;
    if(age<3.0){
      float f=exp(-age*2.0)*smoothstep(0.0,0.05,age);
      flash=f; vNova=1.0; col=mix(col,vec3(0.86,0.93,1.0),min(1.0,f*4.0));
    }
  }
  float df=clamp(d*d/(150.0*150.0),0.12,3.0);
  vCol=col*lum*uFlux*df/(big*big);
  gl_PointSize=uSize*big;
  if(flash>0.0){ vCol=col*uFlux*260.0*flash; gl_PointSize=uSize*(2.0+13.0*sqrt(flash)); }
}
)";

static const char* FS_STARS = R"(
in vec3 vCol; in float vNova; out vec4 o;
void main(){
  vec2 q=gl_PointCoord*2.0-1.0; float d2=dot(q,q);
  float a = vNova>0.5 ? exp(-d2*45.0)+0.07*max(0.0,exp(-d2*4.0)-0.018) : max(0.0,exp(-d2*2.2)-0.11)*1.25;
  o=vec4(vCol*a,1.0);
}
)";

static const char* VS_FULL = R"(
out vec2 uv;
void main(){ vec2 p=vec2((gl_VertexID<<1)&2, gl_VertexID&2); uv=p; gl_Position=vec4(p*2.0-1.0,0.0,1.0); }
)";

static const char* FS_DOWN = R"(
in vec2 uv; out vec4 o; layout(binding=0) uniform sampler2D uTex; uniform vec2 uH;
void main(){
  vec3 s=texture(uTex,uv).rgb*4.0+texture(uTex,uv-uH).rgb+texture(uTex,uv+uH).rgb
        +texture(uTex,uv+vec2(uH.x,-uH.y)).rgb+texture(uTex,uv-vec2(uH.x,-uH.y)).rgb;
  o=vec4(s/8.0,1.0);
}
)";

static const char* FS_UP = R"(
in vec2 uv; out vec4 o; layout(binding=0) uniform sampler2D uTex; uniform vec2 uH;
void main(){
  vec3 s=texture(uTex,uv+vec2(-2.0*uH.x,0.0)).rgb+texture(uTex,uv+vec2(2.0*uH.x,0.0)).rgb
        +texture(uTex,uv+vec2(0.0,2.0*uH.y)).rgb+texture(uTex,uv+vec2(0.0,-2.0*uH.y)).rgb
        +2.0*(texture(uTex,uv+vec2(-uH.x,uH.y)).rgb+texture(uTex,uv+vec2(uH.x,uH.y)).rgb
             +texture(uTex,uv+vec2(uH.x,-uH.y)).rgb+texture(uTex,uv+vec2(-uH.x,-uH.y)).rgb);
  o=vec4(s/12.0,1.0);
}
)";

static const char* FS_FINAL = R"(
in vec2 uv; out vec4 o;
layout(binding=0) uniform sampler2D uHdr; layout(binding=1) uniform sampler2D uBloom;
uniform vec3 uRight, uUp, uFwd; uniform float uTan, uAspect, uExposure, uBloomAmt;
float h13(vec3 p){ p=fract(p*0.1031); p+=dot(p,p.zyx+31.32); return fract((p.x+p.y)*p.z); }
vec3 sky(vec3 d){
  vec3 p=d*140.0, ip=floor(p);
  vec3 r=vec3(h13(ip),h13(ip+17.3),h13(ip+41.7));
  float dist=length(p-(ip+r));
  float b=pow(h13(ip+7.1),9.0)*6.0+0.05;
  vec3 tint=mix(vec3(1.0,0.8,0.65),vec3(0.7,0.82,1.0),h13(ip+3.3));
  return tint*b*smoothstep(0.075,0.0,dist);
}
vec3 tone(vec3 x){   // the stretch astronomers use: faint tails stay visible, bright cores keep their colour
  float l=max(max(x.r,x.g),x.b)+1e-6, lt=asinh(l*8.0)/asinh(8.0*40.0);
  return mix(x*(lt/l), vec3(1.0), smoothstep(12.0,60.0,l)*0.8);
}
void main(){
  vec2 n=uv*2.0-1.0;
  vec3 dir=normalize(uFwd+n.x*uAspect*uTan*uRight+n.y*uTan*uUp);
  vec3 c=texture(uHdr,uv).rgb+uBloomAmt*texture(uBloom,uv).rgb+sky(dir)*0.35;
  c=tone(c*uExposure);
  c*=1.0-0.22*dot(n,n);
  c=pow(c,vec3(1.0/2.2));
  c+=(h13(vec3(gl_FragCoord.xy,1.7))-0.5)/255.0;
  o=vec4(c,1.0);
}
)";

// ---------------------------------------------------------------- GL helpers
static GLuint mkShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  const char* parts[2] = {"#version 460 core\n", src};
  glShaderSource(s, 2, parts, nullptr); glCompileShader(s);
  GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof log, nullptr, log); fprintf(stderr, "shader error:\n%s\n%s\n", log, src); exit(1); }
  return s;
}
static GLuint mkProg(const char* cs) {
  GLuint p = glCreateProgram(); glAttachShader(p, mkShader(GL_COMPUTE_SHADER, cs)); glLinkProgram(p);
  GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, nullptr, log); fprintf(stderr, "link error: %s\n", log); exit(1); }
  return p;
}
static GLuint mkProg(const char* vs, const char* fs) {
  GLuint p = glCreateProgram(); glAttachShader(p, mkShader(GL_VERTEX_SHADER, vs)); glAttachShader(p, mkShader(GL_FRAGMENT_SHADER, fs)); glLinkProgram(p);
  GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, nullptr, log); fprintf(stderr, "link error: %s\n", log); exit(1); }
  return p;
}
static GLint U(GLuint p, const char* n) { return glGetUniformLocation(p, n); }
static GLuint mkTex3D(int size, GLenum fmt, bool linear) {
  GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_3D, t);
  glTexStorage3D(GL_TEXTURE_3D, 1, fmt, size, size, size);
  GLenum f = linear ? GL_LINEAR : GL_NEAREST;
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, f); glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, f);
  for (GLenum w : {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R}) glTexParameteri(GL_TEXTURE_3D, w, GL_CLAMP_TO_BORDER);
  return t;
}
static GLuint mkTex2D(int w, int h, GLenum fmt) {
  GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
  glTexStorage2D(GL_TEXTURE_2D, 1, fmt, w, h);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  return t;
}
static GLuint mkFbo(GLuint tex) {
  GLuint f; glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  return f;
}

// ---------------------------------------------------------------- small vector maths
struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float len(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { return a * (1.0f / len(a)); }

// ---------------------------------------------------------------- state
static GLuint pUpdate, pSpawn, pRhs, pSmooth, pRestrict, pProlong, pGrad, pStars, pDown, pUp, pFinal;
static GLuint bPos, bVel, bCol, bTPos, bTVel, bTCol, vao;
static GLuint tDens, tForce, tRhs[LEVELS], tPhi[LEVELS][2];
static const int BLOOM = 6;
static GLuint tHdr = 0, fHdr = 0, tBloom[BLOOM], fBloom[BLOOM], tLdr = 0, fLdr = 0;
static int fbW = 0, fbH = 0;
static double mUnit;                    // deposit mass unit
static double slotMass[SLOTS];
static float gTime = 0, gNova = 4.0f;
static int firstType = 0, forcedType = -1;
static std::vector<int> typeQueue;
static int thrown = 0;                  // galaxies spawned so far (slot = thrown % SLOTS)
static V3 pendingShift = {0, 0, 0}, kick = {0, 0, 0};
static int extraCycles = 0;
static std::mt19937 rng(7);
static float gFlux = 0.6f, gExposure = 0.3f, gBloom = 0.032f, gBeta = 0.85f, gVel = 0.85f;

// ---------------------------------------------------------------- galaxies: rolled from a recipe, built on the CPU
enum { F_SPIRAL, F_BARRED, F_ELLIPTICAL, F_RING, F_POLAR, F_IRREGULAR, F_SOMBRERO, F_JETS, F_FLOCCULENT, F_GIANT, F_BULLSEYE, F_CARTWHEEL, F_PEA, NFAM };
static double typeMass[NFAM];

// Ring stars are young and bright: each one carries 1/RING_ML of the mass of another star. A ring as heavy
// as it looks is cold and clumps under its own pull within a few turns; a light one keeps its shape.
static const double RING_ML = 4.0;
struct Model {
  double Md = 1.0, Rd = 5.0, z0 = 0.6, Mb = 0.2, ab = 1.2, rmaxB = 12, Mh = 5.0, ah = 14, rmaxH = 45, eps = 1.3, Q = 1.2;
  int nr = 0; double Mr[3] = {0, 0, 0}, Rr[3] = {10, 10, 10}, sr[3] = {1, 1, 1};   // rings: mass, radius, width
  double Mj = 0;                                                                   // jets
  double hern(double M, double a, double r, double rmax) const { r = std::min(r, rmax); return M * r * r / ((r + a) * (r + a)); }
  double menc(double r) const {
    double x = r / Rd, s = Md * (1 - (1 + x) * exp(-x)) + hern(Mb, ab, r, rmaxB) + hern(Mh, ah, r, rmaxH);
    for (int i = 0; i < nr; i++) s += Mr[i] * 0.5 * (1 + erf((r - Rr[i]) / (1.414 * sr[i])));
    return s;
  }
  double grav(double r) const { return GRAV * menc(r) * r / pow(r * r + eps * eps, 1.5); }   // softened pull at radius r
  double vc2(double r) const { return grav(r) * r; }
  double sigma(double R) const { return Md / (2 * M_PI * Rd * Rd) * exp(-R / Rd); }
};
struct Col { float r, g, b; };
static Col mixc(Col a, Col b, double t) { float f = float(std::min(1.0, std::max(0.0, t))); return {a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f}; }
// colour (x brightness) in 3 bytes, then: how strongly it turns young-blue in a crowd, bright knot, dark matter
static uint32_t pack(Col c, double lum, int sens, bool knot, bool halo = false) {
  auto q = [&](float v) { return uint32_t(std::min(255.0, std::max(0.0, v * lum * 127.0 + 0.5))); };
  return q(c.r) | q(c.g) << 8 | q(c.b) << 16 | uint32_t(sens & 3) << 28 | uint32_t(knot) << 30 | uint32_t(halo) << 31;
}
static const Col GOLD{1.0f, 0.60f, 0.28f}, AMBER{1.0f, 0.50f, 0.20f}, CREAM{1.0f, 0.84f, 0.62f}, PALE{1.0f, 0.88f, 0.72f}, BLUE{0.50f, 0.68f, 1.0f},
    ELECTRIC{0.36f, 0.60f, 1.0f}, CYAN{0.38f, 0.88f, 1.0f}, PINK{1.0f, 0.30f, 0.52f}, MAGENTA{1.0f, 0.32f, 0.74f}, VIOLET{0.70f, 0.50f, 1.0f},
    TEAL{0.32f, 1.0f, 0.78f}, WHITE{0.85f, 0.92f, 1.0f}, HAZE{0.62f, 0.66f, 1.0f}, ROSE{1.0f, 0.55f, 0.72f}, EMBER{1.0f, 0.42f, 0.30f}, MINT{0.62f, 0.90f, 0.76f};

// Everything that makes one galaxy itself. roll() fills it at random inside a family, so no two galaxies match.
struct Recipe {
  Model m; int arms = 0, nClump = 0;
  double pitch = 0.38, armFrac = 0, armW = 0.30, armStart = 2.0, armEnd = 1e9, patchy = 0, rbar = 0, flat = 1.0, clumpFrac = 0, Rmax = 32, ringHeat = 0.07;
  Col core = GOLD, disk = PALE, arm = BLUE, knot = PINK, bulgeOut = AMBER, jet = TEAL, clump[3] = {MAGENTA, CYAN, VIOLET}, ringCol[3] = {ELECTRIC, ELECTRIC, ELECTRIC};
  double diskLum = 0.70, armLum = 1.0, bulgeLum = 1.0, bulgeK = 3.0, knotFrac = 0.02, rimLum = 1.0;
  bool polar[3] = {false, false, false}, nucleus = false;
  double armPhase[10] = {0};
};

static Recipe roll(int fam, uint64_t seed) {
  std::mt19937_64 r(seed * 0x9E3779B97F4A7C15ull + 12345);
  auto U = [&](double a, double b) { return a + (b - a) * std::uniform_real_distribution<double>(0, 1)(r); };
  auto I = [&](int a, int b) { return std::min(b, a + int(U(0, 1) * (b - a + 1))); };
  Recipe q; Model& m = q.m;
  struct Pal { Col core, disk, arm, knot; };
  static const Pal pals[] = {{GOLD, PALE, BLUE, PINK}, {GOLD, PALE, BLUE, PINK}, {GOLD, PALE, ELECTRIC, PINK}, {CREAM, HAZE, CYAN, WHITE},
                             {AMBER, {1.0f, 0.70f, 0.45f}, {1.0f, 0.74f, 0.48f}, EMBER}, {PALE, {0.72f, 0.62f, 0.95f}, VIOLET, MAGENTA},
                             {CREAM, MINT, TEAL, WHITE}, {GOLD, {1.0f, 0.76f, 0.70f}, ROSE, PINK}};
  static const Col bright[] = {ELECTRIC, CYAN, MAGENTA, VIOLET, TEAL, PINK, WHITE, ROSE, BLUE};
  Pal p = pals[I(0, 7)];
  q.core = p.core; q.disk = p.disk; q.arm = p.arm; q.knot = p.knot; q.bulgeOut = mixc(p.core, AMBER, 0.5);
  for (double& a : q.armPhase) a = U(0, 2 * M_PI);
  for (Col& c : q.ringCol) c = p.arm;
  for (Col& c : q.clump) c = bright[I(0, 8)];
  double size = U(0.88, 1.12);
  switch (fam) {
    case F_SPIRAL:
      q.arms = U(0, 1) < 0.65 ? 2 : (U(0, 1) < 0.6 ? 3 : 4); q.pitch = U(0.25, 0.50); q.armFrac = U(0.50, 0.70); q.armW = U(0.22, 0.34);
      m.Mb = U(0.10, 0.40); m.ab = U(1.0, 1.5); m.Mh = U(4.5, 5.5); m.Q = U(1.1, 1.3); break;
    case F_BARRED:
      m.Md = 1.2; m.Mb = U(0.05, 0.15); m.ab = 1.0; m.Mh = U(3.2, 4.2); m.Q = 1.05; q.arms = 2; q.armFrac = 0.70; q.rbar = U(5.5, 9.5); q.pitch = U(0.35, 0.55);
      for (double& a : q.armPhase) a = 0;
      if (U(0, 1) < 0.35) { m.nr = 1; m.Mr[0] = 0.18; m.Rr[0] = q.rbar; m.sr[0] = 0.6; }
      break;
    case F_ELLIPTICAL:
      m.Md = 0; m.Mb = U(1.8, 3.6); m.ab = U(3.2, 5.2); m.rmaxB = 40; m.Mh = U(6, 8); m.ah = 18; q.flat = U(0.55, 1.0); size = 1;
      q.core = AMBER; q.bulgeOut = CREAM; q.bulgeK = 10; q.bulgeLum = 0.9;
      if (U(0, 1) < 0.15) { q.core = WHITE; q.bulgeOut = HAZE; }
      break;
    case F_RING:
      m.Md = 0.05; m.Mb = U(0.35, 0.60); m.ab = U(1.3, 1.9); m.rmaxB = 10; q.diskLum = 0.45; q.disk = AMBER;
      m.nr = 1; m.Mr[0] = U(0.25, 0.40); m.Rr[0] = U(11, 17); m.sr[0] = U(1.2, 1.9);
      if (U(0, 1) < 0.30) { m.nr = 2; m.Mr[1] = 0.2; m.Rr[1] = m.Rr[0] * 0.55; m.sr[1] = 0.8; q.ringCol[1] = bright[I(0, 8)]; }
      break;
    case F_POLAR:
      m.Md = U(0.6, 0.8); m.Rd = U(2.8, 3.6); m.z0 = 0.5; m.Mb = 0.3; m.Q = 1.8; q.diskLum = 0.9; q.disk = CREAM;
      m.nr = 1; m.Mr[0] = U(0.3, 0.45); m.Rr[0] = U(11.5, 15.5); m.sr[0] = U(1.2, 1.8); q.polar[0] = true; q.ringCol[0] = U(0, 1) < 0.5 ? CYAN : p.arm;
      break;
    case F_IRREGULAR:
      m.Md = U(0.3, 0.6); m.Rd = U(3.0, 4.2); m.z0 = U(1.0, 1.5); m.Mb = 0; m.Mh = U(1.3, 2.0); m.ah = 9; m.rmaxH = 30; m.Q = 1.5;
      q.nClump = I(5, 12); q.clumpFrac = U(0.35, 0.60); q.disk = HAZE; q.diskLum = 0.55; q.core = HAZE; break;
    case F_SOMBRERO:
      m.Md = U(0.6, 0.8); m.Rd = U(5.0, 6.0); m.z0 = 0.35; m.Mb = U(1.0, 1.6); m.ab = U(2.2, 3.0); m.rmaxB = 28; m.Mh = 6; m.ah = 15; m.Q = 1.6; size = std::min(size, 1.0);
      q.core = GOLD; q.bulgeOut = CREAM; q.bulgeLum = 0.9; q.disk = WHITE; q.diskLum = 0.9; q.rimLum = 1.5; break;
    case F_JETS:
      m.Md = 0.8; m.Rd = U(3.5, 4.5); m.z0 = 0.5; m.Mb = 0.25; m.ab = 1.0; m.Mj = U(0.10, 0.16); q.arms = I(2, 3); q.armFrac = 0.55; q.pitch = U(0.20, 0.30); q.nucleus = true;
      if (U(0, 1) < 0.5) { q.core = PALE; q.disk = {0.72f, 0.62f, 0.95f}; q.arm = VIOLET; q.knot = MAGENTA; }
      { static const Col jets[] = {TEAL, TEAL, CYAN, MAGENTA, WHITE, {1.0f, 0.62f, 0.30f}}; q.jet = jets[I(0, 5)]; }
      break;
    case F_FLOCCULENT:
      q.arms = I(5, 9); q.armW = U(0.13, 0.20); q.armFrac = U(0.50, 0.65); q.pitch = U(0.30, 0.50); q.patchy = 1; m.Mb = U(0.10, 0.25); q.knotFrac = 0.03; break;
    case F_GIANT:
      m.Md = 1.3; m.Rd = U(7.0, 8.0); m.z0 = 0.7; m.Mb = 0.25; m.ab = 1.3; m.Mh = 6; m.ah = 16; m.Q = 1.4; size = 1;
      q.arms = I(2, 4); q.armFrac = 0.45; q.pitch = U(0.30, 0.45); q.diskLum = 0.40; q.armLum = 0.60; break;
    case F_BULLSEYE: {
      m.Md = 0.05; m.Mb = U(0.4, 0.6); m.ab = 1.4; m.rmaxB = 8; q.diskLum = 0.4; size = std::min(size, 1.0);
      static const double RR[3] = {7, 13, 19}, MM[3] = {0.20, 0.28, 0.32}, SS[3] = {0.8, 1.0, 1.2};
      m.nr = I(2, 3); double k = U(0.9, 1.05);
      for (int i = 0; i < m.nr; i++) { m.Mr[i] = MM[i]; m.Rr[i] = RR[i] * k; m.sr[i] = SS[i]; q.ringCol[i] = bright[I(0, 8)]; }
      break; }
    case F_CARTWHEEL:
      m.Md = 0.35; m.Rd = 4.0; m.Mb = 0.2; q.diskLum = 0.5; size = std::min(size, 1.0);
      m.nr = 2; m.Mr[0] = 0.45; m.Rr[0] = U(15, 18); m.sr[0] = 1.3; m.Mr[1] = 0.15; m.Rr[1] = 4.5; m.sr[1] = 0.6; q.ringCol[1] = p.core;
      q.arms = I(5, 8); q.pitch = U(1.0, 1.3); q.armFrac = 0.85; q.armW = 0.08; q.armStart = 4.5; q.armEnd = m.Rr[0];
      q.ringHeat = 0.14; break;   // the spokes shake the rim: a colder rim breaks up
    case F_PEA:
      m.Md = U(0.20, 0.30); m.Rd = U(1.6, 2.1); m.z0 = 0.6; m.Mb = 0.05; m.ab = 0.8; m.rmaxB = 6; m.Mh = 1.0; m.ah = 6; m.rmaxH = 20; m.Q = 1.4; size = 1;
      q.nClump = I(3, 5); q.clumpFrac = 0.30; q.disk = U(0, 1) < 0.6 ? TEAL : CYAN; q.core = WHITE; q.diskLum = 1.3; q.bulgeOut = WHITE;
      for (Col& c : q.clump) c = WHITE;
      break;
  }
  m.Rd *= size; m.ab *= size; m.rmaxB = std::min(40.0, m.rmaxB * size); q.rbar *= size; q.armStart *= size;
  for (int i = 0; i < m.nr; i++) { m.Rr[i] *= size; m.sr[i] *= size; m.Mr[i] /= RING_ML; }
  if (q.rbar > 0) { q.armStart = q.rbar; if (m.nr) m.Rr[0] = q.rbar; }
  if (fam == F_CARTWHEEL) q.armEnd = m.Rr[0];
  return q;
}

// Fills NGAL particles (stars first, dark matter last); returns the galaxy's total mass.
static double buildGalaxy(const Recipe& q, uint64_t seed, float* tp, float* tv, uint32_t* tc) {
  const Model& m = q.m;
  std::mt19937_64 r(seed * 77 + 1000);
  std::uniform_real_distribution<double> ud(1e-9, 1.0 - 1e-9);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto u = [&] { return ud(r); };
  auto g = [&] { return nd(r); };

  // radial tables: potential, and the Jeans velocity spread of each round component
  const int T = 2048; const double r0 = 0.02, r1 = 4000;
  std::vector<double> rr(T), phi(T), sb(T), sh(T);
  for (int i = 0; i < T; i++) rr[i] = r0 * pow(r1 / r0, i / double(T - 1));
  auto rho = [](double s, double a, double rmax) { return s < rmax ? 1.0 / (s * pow(s + a, 3)) : 0.0; };
  double ip = 0, ib = 0, ih = 0; phi[T - 1] = 0; sb[T - 1] = sh[T - 1] = 0;
  for (int i = T - 2; i >= 0; i--) {
    double a = rr[i], b = rr[i + 1], ds = b - a;
    ip += 0.5 * ds * (m.grav(a) + m.grav(b));
    ib += 0.5 * ds * (rho(a, m.ab, m.rmaxB) * m.grav(a) + rho(b, m.ab, m.rmaxB) * m.grav(b));
    ih += 0.5 * ds * (rho(a, m.ah, m.rmaxH) * m.grav(a) + rho(b, m.ah, m.rmaxH) * m.grav(b));
    phi[i] = -ip;
    sb[i] = rho(a, m.ab, m.rmaxB) > 0 ? sqrt(ib / rho(a, m.ab, m.rmaxB)) : 0;
    sh[i] = rho(a, m.ah, m.rmaxH) > 0 ? sqrt(ih / rho(a, m.ah, m.rmaxH)) : 0;
  }
  auto look = [&](const std::vector<double>& t, double x) {
    double f = log(std::min(r1, std::max(x, r0)) / r0) / log(r1 / r0) * (T - 1); int i = std::min(T - 2, int(f)); f -= i;
    return t[i] * (1 - f) + t[i + 1] * f;
  };
  auto omega2 = [&](double R) { return m.vc2(R) / (R * R); };
  auto kappa2 = [&](double R) { double d = 0.01 * R; return std::max(1e-12, R * (omega2(R + d) - omega2(R - d)) / (2 * d) + 4 * omega2(R)); };

  double MdEff = m.Md * 0.9826, MbEff = m.hern(m.Mb, m.ab, m.rmaxB, m.rmaxB), MhEff = m.hern(m.Mh, m.ah, m.rmaxH, m.rmaxH);
  double Mrings = 0; for (int i = 0; i < m.nr; i++) Mrings += m.Mr[i];
  double Ms = MdEff + MbEff + m.Mj + Mrings, Wl = Ms + (RING_ML - 1) * Mrings;   // Wl: the stars' light; a ring star weighs 1/RING_ML of another
  int nRing[3] = {0, 0, 0}, nRings = 0; for (int i = 0; i < m.nr; i++) { nRing[i] = int(NSTAR * RING_ML * m.Mr[i] / Wl); nRings += nRing[i]; }
  int nJet = int(NSTAR * m.Mj / Wl), nBulge = int(NSTAR * MbEff / Wl), nDisk = m.Md > 0 ? NSTAR - nBulge - nRings - nJet : 0;
  nBulge = NSTAR - nDisk - nRings - nJet;
  double mStar = Wl / NSTAR, mRing = mStar / RING_ML, mHalo = MhEff / NHALO;
  double Rref = 2.5 * m.Rd, s0 = m.Md > 0 ? m.Q * 3.36 * GRAV * m.sigma(Rref) / sqrt(kappa2(Rref)) * exp(Rref / (2 * m.Rd)) : 0;
  double Rcut = std::min(6 * m.Rd, q.Rmax);

  size_t k = 0;
  auto put = [&](double x, double y, double z, double mass, double vx, double vy, double vz, uint32_t col) {
    tp[k * 4] = x; tp[k * 4 + 1] = y; tp[k * 4 + 2] = z; tp[k * 4 + 3] = mass;
    tv[k * 4] = vx; tv[k * 4 + 1] = vy; tv[k * 4 + 2] = vz; tv[k * 4 + 3] = 0; tc[k] = col; k++;
  };

  // star-forming clumps (irregular and compact galaxies)
  const int NCL = 12; double clR[NCL], clT[NCL];
  for (int i = 0; i < NCL; i++) { do { clR[i] = -m.Rd * log(u() * u()); } while (clR[i] > 2.8 * m.Rd || clR[i] < 0.3 * m.Rd); clT[i] = 2 * M_PI * u(); }
  double clS = std::min(0.9, 0.26 * m.Rd);

  for (int i = 0; i < nDisk; i++) {
    if (q.nClump > 0 && u() < q.clumpFrac) {
      int c = std::min(q.nClump - 1, int(u() * q.nClump)); double R = clR[c], th = clT[c], vc = sqrt(m.vc2(R)), cs = cos(th), sn = sin(th);
      put(R * cs + clS * g(), R * sn + clS * g(), 0.66 * clS * g(), mStar, -vc * sn + 0.03 * g(), vc * cs + 0.03 * g(), 0.03 * g(),
          pack(u() < 0.5 ? q.clump[c % 3] : mixc(q.clump[c % 3], WHITE, 0.5), 0.75 * std::max(1.0, q.diskLum), 2, u() < 0.06));
      continue;
    }
    double R; do { R = -m.Rd * log(u() * u()); } while (R > Rcut || R < 0.02);
    double th = 2 * M_PI * u(); bool inArm = false, inBar = false;
    if (q.rbar > 0 && R < q.rbar) { if (u() < 0.8) { th = 0.26 * g() + (u() < 0.5 ? M_PI : 0.0); inBar = true; } }
    else if (q.arms > 0 && R > q.armStart && R < q.armEnd) {
      double gate = q.rbar > 0 || q.armStart > 2.5 ? 1.0 : std::min(1.0, std::max(0.0, (R - 0.8 * m.Rd) / (0.9 * m.Rd)));
      int j = std::min(q.arms - 1, int(u() * q.arms));
      if (q.patchy > 0) { double f = R / (1.3 * m.Rd) + q.armPhase[j]; if (f - floor(f) > 0.55) gate = 0; }
      if (u() < q.armFrac * gate) {   // trailing arms
        th = -log(R / q.armStart) / tan(q.pitch) + (q.patchy > 0 ? q.armPhase[j] * 3 : 2 * M_PI * j / q.arms) + q.armW * g(); inArm = true;
      }
    }
    double z = m.z0 * atanh(2 * u() - 1);
    double Om2 = omega2(R), k2 = kappa2(R), vc2 = m.vc2(R);
    double sR = s0 * exp(-R / (2 * m.Rd)), sP = sR * std::min(1.0, std::max(0.5, sqrt(k2 / (4 * Om2)))), sZ = sqrt(M_PI * GRAV * m.sigma(R) * m.z0);
    double vm2 = vc2 + sR * sR * (1 - k2 / (4 * Om2) - 2 * R / m.Rd) - sP * sP;
    double vR = sR * g(), vP = sqrt(std::max(0.0, vm2)) + sP * g(), vZ = sZ * g();
    double c = cos(th), s = sin(th), core = exp(-R / (1.2 * m.Rd));
    uint32_t col;
    if (inBar) col = pack(mixc(q.core, AMBER, 0.4 * u()), 1.0, 0, false);
    else if (inArm) col = u() < q.knotFrac ? pack(q.knot, 1.0, 0, true) : pack(mixc(q.arm, q.core, core * 0.9), q.armLum, 2, false);
    else col = pack(mixc(q.disk, q.core, core), q.diskLum * (R > 2.6 * m.Rd ? q.rimLum : 1.0), 1, false);
    put(R * c, R * s, z, mStar, vR * c - vP * s, vR * s + vP * c, vZ, col);
  }
  auto round = [&](int n, double a, double rmax, const std::vector<double>& sig, double mass, bool halo) {
    for (int i = 0; i < n; i++) {
      double rad; do { double qq = sqrt(u()); rad = a * qq / (1 - qq); } while (rad > rmax);
      double cz = 2 * u() - 1, sz = sqrt(1 - cz * cz), ph = 2 * M_PI * u();
      double s = look(sig, rad), ve2 = -2 * look(phi, rad), vx, vy, vz;
      do { vx = s * g(); vy = s * g(); vz = s * g(); } while (vx * vx + vy * vy + vz * vz > 0.9 * ve2);
      uint32_t col;
      if (halo) col = pack(GOLD, 0, 0, false, true);
      else if (q.nucleus && rad < 0.7 && u() < 0.25) col = pack(WHITE, 1.0, 0, true);
      else col = pack(mixc(q.core, q.bulgeOut, rad / (rad + q.bulgeK)), q.bulgeLum, 0, false);
      double f = halo ? 1.0 : q.flat;
      put(rad * sz * cos(ph), rad * sz * sin(ph), rad * cz * f, mass, vx, vy, vz * f, col);
    }
  };
  round(nBulge, m.ab, m.rmaxB, sb, mStar, false);
  for (int j = 0; j < m.nr; j++)
    for (int i = 0; i < nRing[j]; i++) {   // a ring of young stars on circular orbits
      double R = std::max(0.5, m.Rr[j] + m.sr[j] * g()), th = 2 * M_PI * u(), vc = sqrt(m.vc2(R)), c = cos(th), s = sin(th), d = q.ringHeat * vc;
      double x = R * c, y = R * s, z = 0.35 * g(), vx = -vc * s + d * g(), vy = vc * c + d * g(), vz = d * g();
      uint32_t col = u() < 0.03 ? pack(q.knot, 1.0, 0, true) : pack(mixc(q.ringCol[j], WHITE, 0.25 * u()), 1.0, 2, false);
      if (q.polar[j]) put(x, -z, y, mRing, vx, -vz, vy, col); else put(x, y, z, mRing, vx, vy, vz, col);
    }
  for (int i = 0; i < nJet; i++) {   // two needles: stars swinging straight through the core, up and down the spin axis
    double A = 8 + 26 * u(), z = (0.3 + (A - 0.3) * u()) * (u() < 0.5 ? -1 : 1);
    double v = sqrt(std::max(0.0, 2 * (look(phi, A) - look(phi, fabs(z))))) * (u() < 0.5 ? -1 : 1), w = 0.22 + 0.012 * fabs(z);
    put(w * g(), w * g(), z, mStar, 0, 0, v, pack(mixc(q.jet, WHITE, 0.3 * u()), 1.3, 0, u() < 0.01));
  }
  round(NHALO, m.ah, m.rmaxH, sh, mHalo, true);
  return Ms + MhEff;
}

// Fresh galaxies are rolled on spare CPU threads, so a click never waits for one.
struct Built { int fam; double mass; std::vector<float> tp, tv; std::vector<uint32_t> tc; };
static std::mutex poolMx; static std::condition_variable poolCv; static std::deque<int> poolJobs; static std::deque<Built*> poolDone;
static bool poolQuit = false; static std::atomic<uint64_t> seedNext{1};
static Built* makeOne(int fam, uint64_t s = 0) {
  Built* b = new Built{fam, 0, std::vector<float>(size_t(NGAL) * 4), std::vector<float>(size_t(NGAL) * 4), std::vector<uint32_t>(NGAL)};
  if (!s) s = seedNext++;
  b->mass = buildGalaxy(roll(fam, s), s, b->tp.data(), b->tv.data(), b->tc.data());
  return b;
}
static void poolWorker();
static std::vector<std::thread> poolThreads;
static void stopPool() { { std::lock_guard<std::mutex> l(poolMx); poolQuit = true; } poolCv.notify_all(); for (auto& t : poolThreads) t.join(); poolThreads.clear(); }
static void poolWorker() {
  for (;;) {
    int fam;
    { std::unique_lock<std::mutex> l(poolMx); poolCv.wait(l, [] { return poolQuit || !poolJobs.empty(); }); if (poolQuit) return; fam = poolJobs.front(); poolJobs.pop_front(); }
    Built* b = makeOne(fam);
    { std::lock_guard<std::mutex> l(poolMx); poolDone.push_back(b); }
  }
}

// ---------------------------------------------------------------- gravity solver
static void bindImg(int unit, GLuint tex, GLenum access, GLenum fmt) { glBindImageTexture(unit, tex, 0, GL_TRUE, 0, access, fmt); }
static void run3(int size) { int g = (size + 7) / 8; glDispatchCompute(g, g, g); glMemoryBarrier(GL_ALL_BARRIER_BITS); }
static void smooth(int l, int n) {
  int size = N >> l; float h = float(1 << l);
  glUseProgram(pSmooth); glUniform1i(U(pSmooth, "uSize"), size); glUniform1f(U(pSmooth, "uH2"), h * h);
  bindImg(1, tRhs[l], GL_READ_ONLY, GL_R32F);
  for (int i = 0; i < n; i++) {
    bindImg(0, tPhi[l][i & 1], GL_READ_ONLY, GL_R32F); bindImg(2, tPhi[l][(i + 1) & 1], GL_WRITE_ONLY, GL_R32F);
    run3(size);
  }
}
static void vcycle(int l) {
  if (l == LEVELS - 1) { smooth(l, 24); return; }
  int size = N >> l; float h = float(1 << l);
  smooth(l, 2);
  glUseProgram(pRestrict); glUniform1i(U(pRestrict, "uSize"), size); glUniform1f(U(pRestrict, "uInvH2"), 1.0f / (h * h));
  bindImg(0, tPhi[l][0], GL_READ_ONLY, GL_R32F); bindImg(1, tRhs[l], GL_READ_ONLY, GL_R32F); bindImg(2, tRhs[l + 1], GL_WRITE_ONLY, GL_R32F);
  run3(size / 2);
  float zero = 0; glClearTexImage(tPhi[l + 1][0], 0, GL_RED, GL_FLOAT, &zero);
  vcycle(l + 1);
  glUseProgram(pProlong); glUniform1i(U(pProlong, "uSize"), size);
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, tPhi[l + 1][0]);
  bindImg(0, tPhi[l][0], GL_READ_WRITE, GL_R32F);
  run3(size);
  smooth(l, 2);
}
static void solve(int cycles) {
  glUseProgram(pRhs); glUniform1f(U(pRhs, "uScale"), float(4 * M_PI * GRAV * mUnit));
  bindImg(0, tDens, GL_READ_ONLY, GL_R32UI); bindImg(1, tRhs[0], GL_WRITE_ONLY, GL_R32F);
  run3(N);
  for (int i = 0; i < cycles; i++) vcycle(0);
  glUseProgram(pGrad); glUniform1i(U(pGrad, "uSize"), N);
  bindImg(0, tPhi[0][0], GL_READ_ONLY, GL_R32F); bindImg(1, tForce, GL_WRITE_ONLY, GL_RGBA16F);
  run3(N);
}
static void moveStars(float dt, V3 shift) {
  GLuint zero = 0; glClearTexImage(tDens, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
  glUseProgram(pUpdate);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bPos); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bVel);
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, tForce);
  bindImg(0, tDens, GL_READ_WRITE, GL_R32UI);
  glUniform1f(U(pUpdate, "uDt"), dt); glUniform1f(U(pUpdate, "uUnits"), float(1.0 / mUnit));
  glUniform3f(U(pUpdate, "uShift"), shift.x, shift.y, shift.z); glUniform3f(U(pUpdate, "uKick"), kick.x, kick.y, kick.z);
  glUniform1ui(U(pUpdate, "uCount"), NTOT);
  glDispatchCompute((NTOT + 255) / 256, 1, 1); glMemoryBarrier(GL_ALL_BARRIER_BITS);
  kick = {0, 0, 0};
}
// one simulation step; realDt drives the slow re-centring of the view
static void step(float dt, float realDt) {
  float f = 1.0f - expf(-realDt * 0.7f);
  V3 shift = pendingShift * f; pendingShift = pendingShift - shift;
  moveStars(dt, shift);
  solve(1 + extraCycles); extraCycles = 0;
}

// ---------------------------------------------------------------- galaxies in and out
static void spawn(int type, V3 pos, V3 vel, const float rot[9]) {
  int slot = thrown % SLOTS;
  glUseProgram(pSpawn);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bPos); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bVel);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bTPos); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bTVel);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bCol); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bTCol);
  glUniform1ui(U(pSpawn, "uOffset"), slot * NGAL); glUniform1ui(U(pSpawn, "uCount"), NGAL); glUniform1ui(U(pSpawn, "uTpl"), type * NGAL);
  glUniformMatrix3fv(U(pSpawn, "uRot"), 1, GL_FALSE, rot);
  glUniform3f(U(pSpawn, "uPos"), pos.x + 128, pos.y + 128, pos.z + 128); glUniform3f(U(pSpawn, "uVel"), vel.x, vel.y, vel.z);
  glDispatchCompute((NGAL + 255) / 256, 1, 1); glMemoryBarrier(GL_ALL_BARRIER_BITS);
  slotMass[slot] = typeMass[type];
  double total = 0; for (double mm : slotMass) total += mm;
  float frac = float(typeMass[type] / total);
  kick = kick - vel * frac;                       // keep the whole system at rest
  pendingShift = pendingShift - pos * frac;       // and slide its centre of mass back to the middle
  thrown++; extraCycles = 4;
  { std::lock_guard<std::mutex> l(poolMx); poolJobs.push_back(type); } poolCv.notify_one();   // roll the next one of this family
}
static void upload(Built* b) {
  size_t off = size_t(b->fam) * NGAL;
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bTPos); glBufferSubData(GL_SHADER_STORAGE_BUFFER, off * 16, size_t(NGAL) * 16, b->tp.data());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bTVel); glBufferSubData(GL_SHADER_STORAGE_BUFFER, off * 16, size_t(NGAL) * 16, b->tv.data());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bTCol); glBufferSubData(GL_SHADER_STORAGE_BUFFER, off * 4, size_t(NGAL) * 4, b->tc.data());
  typeMass[b->fam] = b->mass; delete b;
}
static void uploadReady() {
  Built* b = nullptr;
  { std::lock_guard<std::mutex> l(poolMx); if (!poolDone.empty()) { b = poolDone.front(); poolDone.pop_front(); } }
  if (b) upload(b);
}
static void randomRot(float m[9]) {
  std::normal_distribution<float> nd; float q[4], s = 0;
  for (float& c : q) { c = nd(rng); s += c * c; }
  s = 1 / sqrtf(s); float w = q[0] * s, x = q[1] * s, y = q[2] * s, z = q[3] * s;
  float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                2 * (y * z + w * x), 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)};
  memcpy(m, r, sizeof r);
}
static void reset() {
  float z = 0;
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bPos); glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &z);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bVel); glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &z);
  for (int l = 0; l < LEVELS; l++) for (int i = 0; i < 2; i++) glClearTexImage(tPhi[l][i], 0, GL_RED, GL_FLOAT, &z);
  float z4[4] = {0, 0, 0, 0}; glClearTexImage(tForce, 0, GL_RGBA, GL_FLOAT, z4);
  thrown = 0; pendingShift = kick = {0, 0, 0}; for (double& mm : slotMass) mm = 0;
  // first galaxy: disc lying in the x-z plane, tipped a little (columns of a rotation matrix)
  float a = 0.25f, c = cosf(a), s = sinf(a);
  float rot[9] = {c, s, 0, 0, 0, -1, -s, c, 0};
  spawn(firstType, {0, 0, 0}, {0, 0, 0}, rot);
  kick = pendingShift = {0, 0, 0};
  moveStars(0, {0, 0, 0}); solve(14); extraCycles = 0;
}
// throw a galaxy in from direction q (unit vector, seen from the middle); the orbit lies in the plane across `axis`
static void throwGalaxy(V3 q, V3 axis) {
  static std::vector<int> bag;     // every family comes up once before any repeats
  if (bag.empty()) { for (int i = 0; i < NFAM; i++) bag.push_back(i); std::shuffle(bag.begin(), bag.end(), rng); }
  int type;
  if (forcedType >= 0) { type = forcedType; forcedType = -1; }
  else if (!typeQueue.empty()) { type = typeQueue.front(); typeQueue.erase(typeQueue.begin()); }
  else { type = bag.back(); bag.pop_back(); }
  double present = 0; for (int i = 0; i < SLOTS; i++) if (i != thrown % SLOTS) present += slotMass[i];
  float v = gVel * sqrtf(2 * GRAV * float(present + typeMass[type]) / SPAWN_R), beta = gBeta;
  V3 side = norm(cross(axis, q));
  V3 vel = (q * -cosf(beta) + side * sinf(beta)) * v;
  float rot[9]; randomRot(rot);
  spawn(type, q * SPAWN_R, vel, rot);
}

// ---------------------------------------------------------------- camera and drawing
struct Cam { V3 eye, right, up, fwd; float vp[16]; float tanHalf, aspect; };
static Cam camera(float yaw, float pitch, float dist, float aspect) {
  Cam c; c.aspect = aspect; c.tanHalf = tanf(0.5f * 50.0f * PI / 180.0f);
  c.eye = V3{cosf(pitch) * sinf(yaw), sinf(pitch), cosf(pitch) * cosf(yaw)} * dist;
  c.fwd = norm(c.eye * -1.0f); c.right = norm(cross(c.fwd, {0, 1, 0})); c.up = cross(c.right, c.fwd);
  float n = 1.0f, f = 3000.0f, t = 1.0f / c.tanHalf;
  float P[16] = {t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0};
  float V[16] = {c.right.x, c.up.x, -c.fwd.x, 0, c.right.y, c.up.y, -c.fwd.y, 0, c.right.z, c.up.z, -c.fwd.z, 0,
                 -dot(c.right, c.eye), -dot(c.up, c.eye), dot(c.fwd, c.eye), 1};
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { float s = 0; for (int k = 0; k < 4; k++) s += P[k * 4 + j] * V[i * 4 + k]; c.vp[i * 4 + j] = s; }
  return c;
}
static void resizeTargets(int w, int h) {
  if (w == fbW && h == fbH) return;
  if (tHdr) { glDeleteTextures(1, &tHdr); glDeleteFramebuffers(1, &fHdr); glDeleteTextures(BLOOM, tBloom); glDeleteFramebuffers(BLOOM, fBloom); glDeleteTextures(1, &tLdr); glDeleteFramebuffers(1, &fLdr); }
  fbW = w; fbH = h;
  tHdr = mkTex2D(w, h, GL_RGBA16F); fHdr = mkFbo(tHdr);
  for (int i = 0; i < BLOOM; i++) { tBloom[i] = mkTex2D(std::max(1, w >> (i + 1)), std::max(1, h >> (i + 1)), GL_RGBA16F); fBloom[i] = mkFbo(tBloom[i]); }
  tLdr = mkTex2D(w, h, GL_RGBA8); fLdr = mkFbo(tLdr);
}
static void render(const Cam& c, GLuint target) {
  glBindVertexArray(vao);
  // stars, added up in a floating-point picture
  glBindFramebuffer(GL_FRAMEBUFFER, fHdr); glViewport(0, 0, fbW, fbH);
  glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE); glEnable(GL_PROGRAM_POINT_SIZE);
  glUseProgram(pStars);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bPos); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bCol);
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, tRhs[0]);
  glUniform1f(U(pStars, "uTime"), gTime); glUniform1f(U(pStars, "uNova"), gNova);
  float res = fbH / 1440.0f, psize = std::max(1.6f, 2.0f * res);
  glUniformMatrix4fv(U(pStars, "uVP"), 1, GL_FALSE, c.vp);
  glUniform1f(U(pStars, "uSize"), psize);
  glUniform1f(U(pStars, "uFlux"), gFlux * (1048576.0f / NSTAR) * 4.0f / (psize * psize) * (res * res));   // more pixels, fewer stars in each: same glow at any size
  // reference density for "crowded": the disc's mid-plane density at 1.5 scale lengths
  glUniform1f(U(pStars, "uRhoRef"), float(4 * M_PI * GRAV * (1.0 / (2 * M_PI * 25.0)) * exp(-1.5) / 2.0));
  glDrawArrays(GL_POINTS, 0, NTOT);
  glDisable(GL_PROGRAM_POINT_SIZE);
  // bloom: blur down, then back up
  glUseProgram(pDown); glDisable(GL_BLEND);
  for (int i = 0; i < BLOOM; i++) {
    int sw = i ? std::max(1, fbW >> i) : fbW, sh = i ? std::max(1, fbH >> i) : fbH;
    glBindFramebuffer(GL_FRAMEBUFFER, fBloom[i]); glViewport(0, 0, std::max(1, fbW >> (i + 1)), std::max(1, fbH >> (i + 1)));
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, i ? tBloom[i - 1] : tHdr);
    glUniform2f(U(pDown, "uH"), 1.0f / sw, 1.0f / sh);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
  glUseProgram(pUp); glEnable(GL_BLEND);
  for (int i = BLOOM - 2; i >= 0; i--) {
    int sw = std::max(1, fbW >> (i + 2)), sh = std::max(1, fbH >> (i + 2));
    glBindFramebuffer(GL_FRAMEBUFFER, fBloom[i]); glViewport(0, 0, std::max(1, fbW >> (i + 1)), std::max(1, fbH >> (i + 1)));
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tBloom[i + 1]);
    glUniform2f(U(pUp, "uH"), 0.5f / sw, 0.5f / sh);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
  glDisable(GL_BLEND);
  // final picture
  glBindFramebuffer(GL_FRAMEBUFFER, target); glViewport(0, 0, fbW, fbH);
  glUseProgram(pFinal);
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tHdr);
  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tBloom[0]);
  glUniform3f(U(pFinal, "uRight"), c.right.x, c.right.y, c.right.z); glUniform3f(U(pFinal, "uUp"), c.up.x, c.up.y, c.up.z);
  glUniform3f(U(pFinal, "uFwd"), c.fwd.x, c.fwd.y, c.fwd.z);
  glUniform1f(U(pFinal, "uTan"), c.tanHalf); glUniform1f(U(pFinal, "uAspect"), c.aspect);
  glUniform1f(U(pFinal, "uExposure"), gExposure); glUniform1f(U(pFinal, "uBloomAmt"), gBloom);
  glDrawArrays(GL_TRIANGLES, 0, 3);
}

// ---------------------------------------------------------------- input
static double mouseX = 0.5, mouseY = 0.5, zoom = 1.0;
static bool clicked = false, wantReset = false, wantFull = false, paused = false;
static void onCursor(GLFWwindow* w, double x, double y) { int ww, hh; glfwGetWindowSize(w, &ww, &hh); mouseX = x / std::max(1, ww); mouseY = y / std::max(1, hh); }
static void onButton(GLFWwindow*, int b, int action, int) { if (b == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) clicked = true; }
static void onScroll(GLFWwindow*, double, double dy) { zoom = std::min(2.2, std::max(0.35, zoom * pow(0.9, dy))); }
static void onKey(GLFWwindow* w, int key, int, int action, int) {
  if (action != GLFW_PRESS) return;
  if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) glfwSetWindowShouldClose(w, 1);
  if (key == GLFW_KEY_R) wantReset = true;
  if (key == GLFW_KEY_F || key == GLFW_KEY_F11) wantFull = true;
  if (key == GLFW_KEY_SPACE) paused = !paused;
  if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) forcedType = key - GLFW_KEY_1;   // pick the family of the next galaxy
  if (key == GLFW_KEY_0) forcedType = 9;
  if (key == GLFW_KEY_MINUS) forcedType = 10;
  if (key == GLFW_KEY_EQUAL) forcedType = 11;
  if (key == GLFW_KEY_BACKSPACE) forcedType = 12;
}

static void writePPM(const char* path) {
  std::vector<unsigned char> px(size_t(fbW) * fbH * 3);
  glBindFramebuffer(GL_FRAMEBUFFER, fLdr); glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, fbW, fbH, GL_RGB, GL_UNSIGNED_BYTE, px.data());
  FILE* f = fopen(path, "wb"); if (!f) { perror(path); return; }
  fprintf(f, "P6\n%d %d\n255\n", fbW, fbH);
  for (int y = fbH - 1; y >= 0; y--) fwrite(&px[size_t(y) * fbW * 3], 1, size_t(fbW) * 3, f);
  fclose(f);
}
// size of the first galaxy's disc, read back from the GPU (for checking that it holds its shape)
static void stats(int stepNo) {
  const int M = 20000; std::vector<float> p(M * 4);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bPos); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, M * 16, p.data());
  double cx = 0, cy = 0, cz = 0; for (int i = 0; i < M; i++) { cx += p[i * 4]; cy += p[i * 4 + 1]; cz += p[i * 4 + 2]; } cx /= M; cy /= M; cz /= M;
  double r2 = 0; std::vector<double> rs(M);
  for (int i = 0; i < M; i++) { double x = p[i * 4] - cx, y = p[i * 4 + 1] - cy, z = p[i * 4 + 2] - cz; rs[i] = sqrt(x * x + y * y + z * z); r2 += rs[i] * rs[i]; }
  std::sort(rs.begin(), rs.end());
  printf("step %5d  centre %.1f %.1f %.1f  r10 %.2f  r50 %.2f  r90 %.2f  rms %.2f\n", stepNo, cx, cy, cz, rs[M / 10], rs[M / 2], rs[M * 9 / 10], sqrt(r2 / M));
}

int main(int argc, char** argv) {
  const char* shot = nullptr; const char* framesDir = nullptr; int every = 50; int steps = 600, throwAt = -1, throw2 = -1, W = 1600, H = 900, statsEvery = 0; bool test = false, bench = false, windowed = false, fixedSeed = false;
  float sYaw = 0.0f, sPitch = 0.62f, sDist = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i]; auto nx = [&] { return i + 1 < argc ? argv[++i] : (char*)"0"; };
    if (a == "--shot") shot = nx(); else if (a == "--steps") steps = atoi(nx()); else if (a == "--throw") throwAt = atoi(nx());
    else if (a == "--throw2") throw2 = atoi(nx()); else if (a == "--w") W = atoi(nx()); else if (a == "--h") H = atoi(nx());
    else if (a == "--stars") { NSTAR = int(atof(nx()) * 1048576); NHALO = NSTAR / 4; }
    else if (a == "--window") windowed = true;
    else if (a == "--stats") statsEvery = atoi(nx()); else if (a == "--test") test = true; else if (a == "--bench") bench = true;
    else if (a == "--exp") gExposure = atof(nx()); else if (a == "--bloom") gBloom = atof(nx());
    else if (a == "--types") { char* l = nx(); for (char* t = strtok(l, ","); t; t = strtok(nullptr, ",")) typeQueue.push_back(std::max(1, std::min(int(NFAM), atoi(t))) - 1); if (!typeQueue.empty()) { firstType = typeQueue[0]; typeQueue.erase(typeQueue.begin()); } }
    else if (a == "--seed") { seedNext = strtoull(nx(), nullptr, 10); fixedSeed = true; }
    else if (a == "--nova") gNova = atof(nx()); else if (a == "--time") gTime = atof(nx());
    else if (a == "--flux") gFlux = atof(nx());
    else if (a == "--beta") gBeta = atof(nx()); else if (a == "--vel") gVel = atof(nx());
    else if (a == "--frames") framesDir = nx(); else if (a == "--every") every = atoi(nx());
    else if (a == "--yaw") sYaw = atof(nx()); else if (a == "--pitch") sPitch = atof(nx()); else if (a == "--dist") sDist = atof(nx());
    else { printf("galaxies [--window] [--stars millions-per-galaxy]\n  click: throw a galaxy in   mouse: look   wheel: zoom   1-9 0 - = Backspace: family of the next galaxy   R: start over   F: full screen   space: pause   Esc: quit\n"); return 0; }
  }
  NGAL = NSTAR + NHALO; NTOT = NGAL * SLOTS;
  bool headless = shot || test || bench || framesDir;

  if (!glfwInit()) { fprintf(stderr, "no window system\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHintString(GLFW_WAYLAND_APP_ID, "galaxies");
  if (headless) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  bool full = !headless && !windowed;
  GLFWmonitor* mon = glfwGetPrimaryMonitor(); const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
  if (!vm) full = false;
  GLFWwindow* win = full ? glfwCreateWindow(vm->width, vm->height, "galaxies", mon, nullptr) : glfwCreateWindow(W, H, "galaxies", nullptr, nullptr);
  if (!win) { fprintf(stderr, "could not open an OpenGL 4.6 window\n"); return 1; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  glfwSetCursorPosCallback(win, onCursor); glfwSetMouseButtonCallback(win, onButton); glfwSetScrollCallback(win, onScroll); glfwSetKeyCallback(win, onKey);

  pUpdate = mkProg(CS_UPDATE); pSpawn = mkProg(CS_SPAWN); pRhs = mkProg(CS_RHS); pSmooth = mkProg(CS_SMOOTH);
  pRestrict = mkProg(CS_RESTRICT); pProlong = mkProg(CS_PROLONG); pGrad = mkProg(CS_GRAD);
  pStars = mkProg(VS_STARS, FS_STARS); pDown = mkProg(VS_FULL, FS_DOWN); pUp = mkProg(VS_FULL, FS_UP); pFinal = mkProg(VS_FULL, FS_FINAL);
  glGenVertexArrays(1, &vao);

  tDens = mkTex3D(N, GL_R32UI, false); tForce = mkTex3D(N, GL_RGBA16F, true);
  for (int l = 0; l < LEVELS; l++) { tRhs[l] = mkTex3D(N >> l, GL_R32F, true); tPhi[l][0] = mkTex3D(N >> l, GL_R32F, true); tPhi[l][1] = mkTex3D(N >> l, GL_R32F, true); }

  mUnit = (1.2 / NSTAR) / 256.0;
  if (!headless && !fixedSeed) { std::random_device rd; seedNext = (uint64_t(rd()) << 20) | 1; rng.seed(rd()); }
  auto mkBuf = [](size_t bytes, const void* data) { GLuint b; glGenBuffers(1, &b); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_COPY); return b; };
  bTPos = mkBuf(size_t(NFAM) * NGAL * 16, nullptr); bTVel = mkBuf(size_t(NFAM) * NGAL * 16, nullptr); bTCol = mkBuf(size_t(NFAM) * NGAL * 4, nullptr);
  { double t0 = glfwGetTime(); std::vector<Built*> first(NFAM); std::vector<std::thread> th;
    uint64_t base = seedNext; seedNext += NFAM;
    for (int f = 0; f < NFAM; f++) th.emplace_back([&first, f, base] { first[f] = makeOne(f, base + f); });
    for (auto& x : th) x.join();
    for (Built* b : first) upload(b);
    if (headless) printf("built %d galaxies in %.2f s\n", NFAM, glfwGetTime() - t0); }
  poolThreads.emplace_back(poolWorker); poolThreads.emplace_back(poolWorker); atexit(stopPool);
  bPos = mkBuf(size_t(NTOT) * 16, nullptr); bVel = mkBuf(size_t(NTOT) * 16, nullptr); bCol = mkBuf(size_t(NTOT) * 4, nullptr);

  if (test) {   // one point mass in the middle: the solved pull must follow 1/r^2
    float z = 0; glBindBuffer(GL_SHADER_STORAGE_BUFFER, bPos); glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &z);
    float one[4] = {128.5f, 128.5f, 128.5f, 1.0f}; glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, one);
    moveStars(0, {0, 0, 0});
    for (int c : {1, 1, 2, 4, 8, 16}) {
      solve(c); float f[64 * 4]; glGetTextureSubImage(tForce, 0, 128, 128, 128, 64, 1, 1, GL_RGBA, GL_FLOAT, sizeof f, f);
      printf("after +%d cycles:", c); for (int r : {2, 4, 8, 16, 32, 60}) printf("  r=%d ratio %.3f", r, -f[r * 4] * r * r / GRAV); printf("\n");
    }
    return 0;
  }

  reset();
  float yaw = sYaw, pitch = sPitch, dist = 120, sm = 0.5f, smy = 0.5f;

  if (shot || bench || framesDir) {
    resizeTargets(W, H);
    double t0 = glfwGetTime();
    for (int s = 0; s < steps; s++) {
      if (s == throwAt) throwGalaxy(norm(V3{0.8f, 0.45f, -0.4f}), norm(V3{0.2f, 0.9f, 0.4f}));
      if (s == throw2) throwGalaxy(norm(V3{-0.7f, -0.2f, 0.6f}), norm(V3{0.2f, 0.9f, 0.4f}));
      uploadReady();
      step(0.5f, 1.0f / 60); gTime += 1.0f / 60;
      if (statsEvery && s % statsEvery == 0) stats(s);
      if (framesDir && s % every == 0) {
        dist += ((thrown > 1 ? 200.0f : 120.0f) - dist) * 0.02f * every / 4;
        char path[512]; snprintf(path, sizeof path, "%s/f%05d.ppm", framesDir, s / every);
        render(camera(sYaw + s * 0.02f / 60, sPitch, sDist > 0 ? sDist : dist, float(W) / H), fLdr); writePPM(path);
      }
    }
    glFinish(); double t1 = glfwGetTime();
    printf("%d steps, %.2f ms per step, %d particles\n", steps, (t1 - t0) * 1000 / std::max(1, steps), NTOT);
    Cam c = camera(sYaw, sPitch, sDist > 0 ? sDist : (thrown > 1 ? 200 : 120), float(W) / H);
    if (bench) { double r0 = glfwGetTime(); for (int i = 0; i < 100; i++) render(c, fLdr); glFinish(); printf("%.2f ms per drawn frame at %dx%d\n", (glfwGetTime() - r0) * 10, W, H); }
    if (shot) { render(c, fLdr); writePPM(shot); }
    return 0;
  }

  double last = glfwGetTime(), start = last, fpsT = last; int frames = 0; bool everClicked = false;
  int winX = 0, winY = 0, winW = W, winH = H;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    double now = glfwGetTime(); float rdt = float(std::min(0.05, now - last)); last = now;
    int w, h; glfwGetFramebufferSize(win, &w, &h); if (w < 8 || h < 8) { glfwWaitEventsTimeout(0.1); continue; }
    resizeTargets(w, h);
    if (wantFull) {
      wantFull = false; full = !full;
      if (full && vm) { glfwGetWindowPos(win, &winX, &winY); glfwGetWindowSize(win, &winW, &winH); glfwSetWindowMonitor(win, mon, 0, 0, vm->width, vm->height, vm->refreshRate); }
      else glfwSetWindowMonitor(win, nullptr, winX, winY, winW, winH, 0);
    }
    if (wantReset) { wantReset = false; reset(); start = now; everClicked = false; }

    float k = 1.0f - expf(-rdt * 4.0f); sm += (float(mouseX) - sm) * k; smy += (float(mouseY) - smy) * k;
    yaw = float(now - start) * 0.02f + (sm - 0.5f) * 4.0f;
    pitch = std::min(1.45f, std::max(-1.45f, 0.62f + (smy - 0.5f) * 2.2f));
    float want = (thrown > 1 ? 200.0f : 120.0f) * float(zoom); dist += (want - dist) * (1.0f - expf(-rdt * 0.8f));
    Cam c = camera(yaw, pitch, dist, float(w) / h);

    if (clicked) {
      clicked = false; everClicked = true;
      float nx = float(mouseX) * 2 - 1, ny = 1 - float(mouseY) * 2;
      V3 q = c.right * (nx * c.aspect) + c.up * ny;          // click direction, flat in the screen plane
      throwGalaxy(len(q) < 0.02f ? c.right : norm(q), c.fwd);
    }
    if (!everClicked && thrown == 1 && now - start > 8.0) { everClicked = true; throwGalaxy(norm(c.right * 0.8f + c.up * 0.5f), c.fwd); }

    uploadReady();
    if (!paused) { step(std::min(rdt * SPEED, 0.9f), rdt); gTime += rdt; }
    render(c, 0);
    glfwSwapBuffers(win);
    if (++frames, now - fpsT > 3.0) { if (getenv("GALAXIES_FPS")) { printf("%.0f fps  %dx%d\n", frames / (now - fpsT), w, h); fflush(stdout); } frames = 0; fpsT = now; }
  }
  glfwTerminate();
  return 0;
}
