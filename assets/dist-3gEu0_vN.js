import{n as e}from"./index-D5W70bXr.js";import{$a as t,$n as n,$o as r,$s as i,Ac as a,An as o,Ao as s,As as c,At as l,Ba as u,Bc as d,Bi as f,Bo as p,Bs as m,Bt as h,Ca as g,Cc as _,Cn as v,Co as y,Ct as b,Da as x,Do as S,Ds as C,Dt as w,Ea as T,Ec as E,Eo as ee,Es as D,Et as O,Fa as te,Fc as ne,Fo as re,Fs as ie,Gc as ae,Gt as oe,Ha as se,Hc as ce,Ho as le,Hs as ue,Ht as de,Ia as fe,Io as pe,Is as me,Ja as he,Js as ge,Jt as _e,Ka as ve,Ki as ye,Ks as be,Kt as xe,La as Se,Ln as Ce,Lo as we,Ls as Te,Lt as Ee,Mc as k,Mo as De,Mr as Oe,Ms as ke,Na as Ae,Ns as je,Oa as Me,Oo as Ne,Os as Pe,Ot as Fe,Pa as Ie,Po as Le,Ps as Re,Qa as ze,Qn as Be,Qo as Ve,Qt as He,Ra as Ue,Rc as We,Ro as Ge,Rs as Ke,Sa as qe,Sc as Je,So as Ye,St as Xe,Ta as Ze,Tc as Qe,Tn as $e,To as et,Ts as tt,Tt as nt,Ua as rt,Uo as it,Us as at,Ut as ot,Va as st,Vc as ct,Vs as lt,Vt as ut,Wa as dt,Wc as ft,Wo as pt,Ws as mt,Wt as ht,Xa as gt,Xc as _t,Xo as vt,Xs as yt,Xt as bt,Ya as xt,Yc as St,Yi as Ct,Yo as wt,Ys as Tt,Yt as Et,Za as Dt,Zo as Ot,Zt as kt,_a as At,_c as jt,_o as Mt,_r as Nt,aa as Pt,ac as Ft,ai as It,ao as Lt,ba as Rt,bc as zt,bo as Bt,br as Vt,bs as Ht,cc as Ut,co as Wt,cr as Gt,da as Kt,dl as qt,ds as Jt,ec as Yt,el as Xt,en as Zt,eo as Qt,er as $t,es as en,fl as tn,fo as nn,fs as rn,ga as an,gc as on,go as sn,gr as cn,gs as ln,hc as un,ho as dn,hr as A,hs as fn,ic as pn,in as mn,io as hn,is as gn,ja as _n,jo as vn,js as yn,ka as bn,kc as j,ko as xn,ks as Sn,kt as Cn,lc as wn,ll as Tn,lo as En,ls as Dn,ms as On,nc as kn,nl as An,nn as jn,no as Mn,nr as Nn,ol as M,on as Pn,oo as Fn,or as N,os as In,pa as Ln,pi as Rn,po as zn,ps as Bn,qa as Vn,qi as Hn,qs as Un,qt as Wn,rc as Gn,ro as Kn,rr as qn,rs as Jn,sc as Yn,sn as Xn,ss as Zn,t as Qn,tc as $n,tn as er,to as tr,tr as nr,ts as rr,ua as ir,un as ar,va as or,vc as sr,vo as cr,vr as lr,wa as ur,wo as dr,ws as fr,xa as pr,xo as mr,xr as hr,xt as gr,ya as _r,yc as vr,yo as yr,yr as br,ys as xr,za as Sr,zc as Cr,zi as wr,zs as Tr,zt as Er}from"./shared-BOO_KCas.js";var P=j();P.registerFlag(`WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE`,()=>15),P.registerFlag(`WEBGPU_CPU_FORWARD`,()=>!0),P.registerFlag(`WEBGPU_MATMUL_PROGRAM_TYPE`,()=>-1),P.registerFlag(`WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE`,()=>!0),P.registerFlag(`WEBGPU_USE_LOW_POWER_GPU`,()=>!1),P.registerFlag(`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`,()=>1e3),P.registerFlag(`WEBGPU_USE_PROFILE_TOOL`,()=>!1),P.registerFlag(`WEBGPU_IMPORT_EXTERNAL_TEXTURE`,()=>!0),P.registerFlag(`WEBGPU_USE_NAIVE_CONV2D_DEBUG`,()=>!1),P.registerFlag(`WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL`,()=>-1),P.registerFlag(`WEBGPU_CONV_SEPARATE_IM2COL_SHADER`,()=>!1),P.registerFlag(`WEBGPU_PRINT_SHADER`,()=>``),P.registerFlag(`WEBGPU_ENGINE_COMPILE_ONLY`,()=>!1);var Dr=class{constructor(e){e&&(this.vendor=e.vendor,this.architecture=e.architecture,this.intelGPUGeneration=this.getIntelGPUGeneration())}getIntelGPUGeneration(){if(this.isIntel()){if(this.architecture.startsWith(`gen`))return Number(this.architecture.match(/\d+/));if(this.architecture.startsWith(`xe`))return 12}return 0}isIntel(){return this.vendor===`intel`}},Or=class{constructor(e){this.device=e,this.numUsedBuffers=0,this.numFreeBuffers=0,this.freeBuffers=new Map,this.usedBuffers=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireBuffer(e,t,n=!1,r=!0){let i,a=kr(e,t);return r?(this.freeBuffers.has(a)||this.freeBuffers.set(a,[]),this.freeBuffers.get(a).length>0?(i=this.freeBuffers.get(a).pop(),this.numFreeBuffers--):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e)):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e),this.usedBuffers.has(a)||this.usedBuffers.set(a,[]),this.usedBuffers.get(a).push(i),this.numUsedBuffers++,this.numBytesUsed+=e,i}releaseBuffer(e,t=!0){if(this.freeBuffers.size===0)return;let n=e.size,r=e.usage,i=kr(n,r),a=this.usedBuffers.get(i),o=a.indexOf(e);if(o<0)throw Error(`Cannot find the buffer in buffer manager`);a[o]=a[a.length-1],a.pop(),this.numUsedBuffers--,this.numBytesUsed-=n,t?(this.freeBuffers.get(i).push(e),this.numFreeBuffers++):(e.destroy(),this.numBytesAllocated-=n)}getNumUsedBuffers(){return this.numUsedBuffers}getNumFreeBuffers(){return this.numFreeBuffers}dispose(){this.freeBuffers.forEach((e,t)=>{e.forEach(e=>{e.destroy()})}),this.usedBuffers.forEach((e,t)=>{e.forEach(e=>{e.destroy()})}),this.freeBuffers=new Map,this.usedBuffers=new Map,this.numUsedBuffers=0,this.numFreeBuffers=0,this.numBytesUsed=0,this.numBytesAllocated=0}};function kr(e,t){return`${e}_${t}`}var Ar=class{constructor(e){this.device=e,this.numUsedTextures=0,this.numFreeTextures=0,this.freeTextures=new Map,this.usedTextures=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireTexture(e,t,n,r){let i=Mr(n),a=e*t*i,o=jr(e,t,n,r);if(this.freeTextures.has(o)||this.freeTextures.set(o,[]),this.usedTextures.has(o)||this.usedTextures.set(o,[]),this.numBytesUsed+=a,this.numUsedTextures++,this.freeTextures.get(o).length>0){this.numFreeTextures--;let e=this.freeTextures.get(o).shift();return this.usedTextures.get(o).push(e),e}this.numBytesAllocated+=a;let s=this.device.createTexture({size:[e,t],format:n,usage:r});return this.usedTextures.get(o).push(s),s}releaseTexture(e){if(this.freeTextures.size===0)return;let t=e.width,n=e.height,r=e.format,i=e.usage,a=jr(t,n,r,i);this.freeTextures.has(a)||this.freeTextures.set(a,[]),this.freeTextures.get(a).push(e),this.numFreeTextures++,this.numUsedTextures--;let o=this.usedTextures.get(a),s=o.indexOf(e);if(s<0)throw Error(`Cannot release a texture that was never provided by this texture manager`);o.splice(s,1);let c=Mr(r),l=t*n*c;this.numBytesUsed-=l}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){this.freeTextures.forEach((e,t)=>{e.forEach(e=>{e.destroy()})}),this.usedTextures.forEach((e,t)=>{e.forEach(e=>{e.destroy()})}),this.freeTextures=new Map,this.usedTextures=new Map,this.numUsedTextures=0,this.numFreeTextures=0,this.numBytesUsed=0,this.numBytesAllocated=0}};function jr(e,t,n,r){return`${e}_${t}_${n}_${r}`}function Mr(e){if(e===`rgba8unorm`)return 16;throw Error(`${e} is not supported!`)}function Nr(e,t){if(Math.max(...e)>5)throw Error(`Cannot symbolically compute strides for rank > 6 tensor.`);let n=e.length,r=e.map(e=>`${t}.${`xyzwuv`[e]}`),i=Array(n-1);i[n-2]=r[n-1];for(let e=n-3;e>=0;--e)i[e]=`(${i[e+1]} * ${r[e+1]})`;return i}var F=(e,t,n)=>n===`int32`?`atomicAdd(${e}, bitcast<i32>(${t}));`:`
          {
            var oldValue = 0;
            loop {
              let newValueF32 = bitcast<f32>(oldValue) + (${t});
              let newValue = bitcast<i32>(newValueF32);
              let res = atomicCompareExchangeWeak(${e}, oldValue, newValue);
              if res.exchanged {
                break;
              }
              oldValue = res.old_value;
            }
          }`,Pr;(function(e){e[e.FROM_PIXELS=0]=`FROM_PIXELS`,e[e.DRAW=1]=`DRAW`})(Pr||={});var Fr=(e,t,n,r,i)=>{let a=Rr(n,{dtype:r.dtype,shape:r.shape},t),o=e.createShaderModule({code:a,label:t.constructor.name}),s=j().get(`WEBGPU_PRINT_SHADER`);if(s!==``){s=s.toLowerCase();let e=s.split(`,`);(s===`all`||e.some(e=>t.shaderKey.toLowerCase().includes(e)))&&(console.group(t.shaderKey),console.debug(a),console.groupEnd())}return i?e.createComputePipelineAsync({compute:{module:o,entryPoint:`_start`},label:t.constructor.name,layout:`auto`}):e.createComputePipeline({compute:{module:o,entryPoint:`_start`},label:t.constructor.name,layout:`auto`})},I=(e,t=`f32`)=>{switch(e){case 1:return`${t}`;case 2:return`vec2<${t}>`;case 3:return`vec3<${t}>`;case 4:return`vec4<${t}>`;default:throw Error(`${e}-component ${t} is not supported.`)}};function L(e){if(e<=1)return`i32`;if(e===2)return`vec2<i32>`;if(e===3)return`vec3<i32>`;if(e===4)return`vec4<i32>`;if(e===5)return`vec5`;if(e===6)return`vec6`;throw Error(`GPU for rank ${e} is not yet supported`)}function R(e){if(e===0)return`x`;if(e===1)return`y`;if(e===2)return`z`;if(e===3)return`w`;if(e===4)return`u`;if(e===5)return`v`;throw Error(`Index ${e} is not yet supported`)}function z(...e){let t;switch(e.length){case 0:t=`
        fn main()
      `;break;case 1:t=`
        fn main(${e[0]} : i32)
      `;break;default:throw Error(`Unreachable`)}return t}function Ir(e,t){let n;return n=`
     ${Lr(t)}
      fn _start(@builtin(local_invocation_id) LocalId : vec3<u32>,
                @builtin(global_invocation_id) GlobalId : vec3<u32>,
                @builtin(local_invocation_index) LocalIndex: u32,
                @builtin(workgroup_id) WorkgroupId : vec3<u32>,
                @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
        localId = LocalId;
        localIndex = LocalIndex;
        globalId = GlobalId;
        numWorkgroups = NumWorkgroups;
        workgroupId = WorkgroupId;
        ${e?`main(getGlobalIndex());`:`main();`};
      }
    `,n}function Lr(e){return`
  @compute @workgroup_size(${e.workgroupSize[0]}, ${e.workgroupSize[1]}, ${e.workgroupSize[2]})
`}function Rr(e,t,n){let r=[],i=n.workgroupSize[0]*n.workgroupSize[1]*n.workgroupSize[2];if(n.outputComponent=n.outputComponent?n.outputComponent:1,r.push(`

      var<private> localId: vec3<u32>;
      var<private> localIndex: u32;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;
      var<private> workgroupId: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
        ${Jr(n)?`  return i32(globalId.x);`:`  return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
                workgroupId.y * numWorkgroups.x + workgroupId.x) * ${i}u +
                localIndex);
        `}
      }
    `),n.pixelsOpType!=null){let i=n.pixelsOpType===Pr.FROM_PIXELS?`@group(0) @binding(0) var<storage, read_write> result: array<${Yr(t.dtype,n.outputComponent)}>;`:`@group(0) @binding(1) var<storage, read> inBuf : array<${Yr(e[0].dtype,n.outputComponent)}>;`,a=t.shape.length===3?`vec2<i32>`:`i32`;r.push(`
        struct Uniform {
          outShapeStrides : ${a},
          size            : i32,
          numChannels     : i32,
          alpha           : f32,
        };

        ${i}
        @group(0) @binding(2) var<uniform> uniforms: Uniform;
      `);let o=Qr(n);return[Br,r.join(`
`),Hr(t.shape),n.getUserCode(),Ir(o,n)].join(`
`)}let a,o,s=`struct Uniforms { NAN : f32, INFINITY : f32, `;n.variableNames.forEach((t,n)=>{let r=L(e[n].shape.length);s+=`${t.charAt(0).toLowerCase()+t.slice(1)}Shape : ${r}, `,a=e[n].shape.length-1,o=L(a),s+=`${t.charAt(0).toLowerCase()+t.slice(1)}ShapeStrides: ${o}, `});let c=L(t.shape.length);s+=`outShape : ${c}, `,a=t.shape.length-1,o=L(a),s+=`
         outShapeStrides: ${o}, `,n.size&&(s+=`size : i32, `),n.uniforms&&(s+=n.uniforms),s+=`};`,s=Zr(s),r.push(s),n.atomic?r.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<atomic<i32>>;
    `):r.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<${Yr(t.dtype,n.outputComponent)}>;
    `),n.variableNames.forEach((t,i)=>{r.push(`
      @group(0) @binding(${1+i}) var<storage, read> ${t}: array<${n.variableComponents?Yr(e[i].dtype,n.variableComponents[i]):Yr(e[i].dtype,n.outputComponent)}>;
        `)}),s!==``&&r.push(`
      @group(0) @binding(${1+n.variableNames.length}) var<uniform> uniforms: Uniforms;
      `);let l=Kr(t.shape,n.dispatchLayout),u=[Br,r.join(`
`)+Vr,Hr(t.shape),l,qr(t.shape.length)];n.atomic||u.push(Xr(t.shape,t.dtype,n.outputComponent)),n.variableNames.forEach((t,n)=>{u.push(`${Hr(e[n].shape,t)}`)});let d=e.map((e,r)=>Gr(e,t.shape,n.variableComponents?n.variableComponents[r]:n.outputComponent,n.dispatchLayout.x.length===t.shape.length)).join(`
`);u.push(d),u.push(n.getUserCode());let f=Qr(n);return u.push(Ir(f,n)),u.join(`
`)}function zr(e,t,n){let r=e.shaderKey;if(e.pixelsOpType!=null)return r;let i=[],o=[];t.forEach(e=>{i.push(e.shape),o.push(e.dtype)}),i.push(n.shape),o.push(n.dtype);let s=t.map(e=>Gt(e.shape,n.shape)),c=t.map(e=>a(e.shape,n.shape)).join(`_`),l=s.map(e=>e.join(`_`)).join(`;`),u=Jr(e)?`flatDispatch`:``;return r+=`_`+(e.workgroupSize?e.workgroupSize.join(`,`):``)+i.map(e=>e.length).join(`,`)+o.join(`,`)+e.variableNames.join(`,`)+l+c+u,r}var Br=`
  struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
  struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) && all(coord < shape);
  }

  fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
    return coord;
  }
  fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return dot(coords, vec2<i32>(shape.y, 1));
  }
  fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
  }
  fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return dot(coords, vec4<i32>(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
  }
  fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
    let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
  }
  fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
    let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
  }

  // NaN defination in IEEE 754-1985 is :
  //   - sign = either 0 or 1.
  //   - biased exponent = all 1 bits.
  //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
  // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
  fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
  }
  fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
    let floatToUint: vec4<u32> = bitcast<vec4<u32>>(val);
    return (floatToUint & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
  }
`,Vr=`
  fn isinf(val: f32) -> bool {
    return abs(val) == uniforms.INFINITY;
  }
`;function Hr(e,t=``){let n=e.length,r=t===``?`getCoordsFromIndex`:`get${t.charAt(0).toUpperCase()+t.slice(1)}CoordsFromIndex`,i=t===``?`outShapeStrides`:`${t.charAt(0).toLowerCase()+t.slice(1)}ShapeStrides`;if(n<=1)return`fn ${r}(index : i32) -> i32 { return index; }`;let a=We(e),o=L(n),s=[];for(let e=0;e<n;e++)s.push(`d${e}`);if(a.length===1)return`    fn ${r}(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.${i}; let d1 = index - d0 * uniforms.${i};
      return vec2<i32>(d0, d1);
    }`;let c;return c=`var index2 = index;`+a.map((e,t)=>`${`let ${s[t]} = index2 / uniforms.${i}.${R(t)}`}; ${t===a.length-1?`let ${s[t+1]} = index2 - ${s[t]} * uniforms.${i}.${R(t)}`:`index2 = index2 - ${s[t]} * uniforms.${i}.${R(t)}`};`).join(``),`
    fn ${r}(index : i32) -> ${o} {
      ${c}
      return ${o}(${s.join(`,`)});
    }
  `}function Ur(e,t){let n=e.name,r=e.shape.length,i=L(r),a=`get`+n.charAt(0).toUpperCase()+n.slice(1),o=[`d0`,`d1`,`d2`,`d3`,`d4`,`d5`].slice(0,r),s=o.map(e=>`${e} : i32`).join(`, `);if(r<1)return`
      fn ${a}() -> ${I(t)} {
        return ${I(t)}(${n}[0]);
      }
    `;let c=`uniforms.${n.charAt(0).toLowerCase()+n.slice(1)}Shape`,l=`${r}D`;return r===0&&(l=`1D`),`
    fn ${a}(${s}) -> ${I(t)} {
      return ${I(t)}(${n}[getIndexFromCoords${l}(${i}(${o.join(`,`)}),
        ${c})${t===1?``:` / ${t}`}]);
    }
   `}function Wr(e,t,n,r){let i=e.name,o=i.charAt(0).toUpperCase()+i.slice(1),s=`get`+o+`ByOutput`,c=e.shape.length,l=t.length,u=L(l);if(a(e.shape,t)&&r)return`
    fn ${s}Index(globalIndex : i32) -> ${I(n)} {
      return ${I(n)}(${i}[globalIndex]);
    }

    fn ${s}Coords(coords : ${u}) -> ${I(n)} {
      return ${I(n)}(${i}[${l>1?`getOutputIndexFromCoords(coords)`:`coords`}${n===1?``:` / ${n}`}]);
    }
    `;let d=Gt(e.shape,t),f=l-c,p=``;if(c===0)return`
    fn ${s}Index(globalIndex : i32) -> ${I(n)}{
      return get${o}();
    }

    fn ${s}Coords(coords : ${u}) -> ${I(n)}{
      return get${o}();
    }
  `;p=l<2&&d.length>=1?`coords = 0;`:d.map(e=>`coords.${R(e+f)} = 0;`).join(`
`);let m=``;m=l<2&&c>0?`coords`:l>1?`${L(c)}(${e.shape.map((e,t)=>`coords.${R(t+f)}`).join(`, `)})`:`coords`;let h=`uniforms.${i.charAt(0).toLowerCase()+i.slice(1)}Shape`,g=`${c}D`;return`
  fn ${s}Index(globalIndex : i32) -> ${I(n)} {
    var coords = getCoordsFromIndex(globalIndex);
    ${p}
    return ${I(n)}(${i}[getIndexFromCoords${g}(${m}, ${h})${n===1?``:` / ${n}`}]);
  }

  fn ${s}Coords(coordsIn : ${u}) -> ${I(n)} {
    var coords = coordsIn;
    ${p}
    return ${I(n)}(${i}[getIndexFromCoords${g}(${m}, ${h})${n===1?``:` / ${n}`}]);
  }
`}function Gr(e,t,n,r){let i=Ur(e,n);return e.shape.length<=t.length&&(i+=Wr(e,t,n,r)),i}function Kr(e,t){let{x:n,y:r=[],z:i=[]}=t,a=e.length,o=n.length+r.length+i.length;if(o!==a)return``;if(n.length===a)return`fn getOutputCoords() -> ${L(a)}{
    let globalIndex = getGlobalIndex();
    return getCoordsFromIndex(globalIndex);
  }
  `;let s=``,c=[n,r,i];for(let e=0;e<c.length;e++){let t=c[e];if(t.length!==0){if(t.length===1)s+=`let d${t[0]} = i32(globalId[${e}]);`;else{let n=Nr(t,`uniforms.outShape`);s+=`var index${e} = i32(globalId[${e}]);`;for(let r=0;r<n.length;r++)s+=`let d${t[r]} = index${e} / ${n[r]};`,r===n.length-1?s+=`let d${t[r+1]} = index${e} - d${t[r]} * ${n[r]};`:s+=`index${e} = index${e} - d${t[r]} * ${n[r]};`}}}let l=[];for(let e=0;e<o;e++)l.push(`d${e}`);let u=L(o),d=`fn getOutputCoords() -> ${u} {
  ${s}
`;return l.length===0?d+=`return ${u}(0); }`:d+=`return ${u}(${l.join(`,`)}); }`,d}function qr(e){let t=``;switch(e){case 0:case 1:t+=`
        fn getOutputIndexFromCoords(coords : i32) -> i32 {
          return coords;
        }
        `;break;case 2:t+=`
        fn getOutputIndexFromCoords(coords : vec2<i32>) -> i32 {
          return dot(coords, vec2<i32>(uniforms.outShapeStrides, 1));
        }
        `;break;case 3:t+=`
        fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
          return dot(coords, vec3<i32>(uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, 1));
        }
        `;break;case 4:t+=`
        fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
          return dot(coords, vec4<i32>(
            uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, uniforms.outShapeStrides.z, 1));
        }
        `;break;case 5:t+=`
        fn getOutputIndexFromCoords(coords : vec5) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u;
        }
        `;break;case 6:t+=`
        fn getOutputIndexFromCoords(coords : vec6) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u * uniforms.outShapeStrides.u +
              coords.v;
        }
        `;break;default:k(!1,()=>`Unsupported ${e}D shape`)}return t}function Jr(e){return e.dispatch[1]===1&&e.dispatch[2]===1}function Yr(e,t=1){if(e===`float32`)return I(t,`f32`);if(e===`int32`||e===`bool`)return I(t,`i32`);throw Error(`type ${e} is not supported.`)}function Xr(e,t,n){let r=e.length,i=Yr(t,n),a=`fn setOutputAtIndex(flatIndex : i32, value : ${I(n)}) {
      result[flatIndex] = ${i}(value);
    }

    fn setOutputAtIndexI32(flatIndex : i32, value : ${I(n,`i32`)}) {
      result[flatIndex] = ${i}(value);
    }
    `;if(r>=2){let e=[`d0`,`d1`,`d2`,`d3`,`d4`,`d5`].slice(0,r),t=L(r);a+=`
      fn setOutputAtCoords(${e.map(e=>`${e} : i32`).join(`, `)}, value : ${I(n)}) {
        let flatIndex = getOutputIndexFromCoords(${t}(${e.join(`, `)}));
        setOutputAtIndex(flatIndex${n===1?``:` / ${n}`}, value);
      }
      fn setOutputAtCoordsI32(${e.map(e=>`${e} : i32`).join(`, `)}, value : ${I(n,`i32`)}) {
        let flatIndex = getOutputIndexFromCoords(${t}(${e.join(`, `)}));
        setOutputAtIndexI32(flatIndex${n===1?``:` / ${n}`}, value);
      }
    `}return a}function Zr(e){return e=e.replace(/(\w+)\s*:\s*vec(5|6)/g,e=>`@align(16) `+e),e=e.replace(/vec(5|6)\s*,\s*(\w+)/g,(e,t,n)=>`vec${t}, @align(16) ${n}`),e}function Qr(e){return!(e.dispatchLayout.hasOwnProperty(`y`)&&e.dispatchLayout.y.length!==0||e.dispatchLayout.hasOwnProperty(`z`)&&e.dispatchLayout.z.length!==0)}var $r=e({GPUBytesPerElement:()=>ai,MatMulProgramType:()=>H,assertNotComplex:()=>si,computeDispatch:()=>B,computeWorkPerThreadForConv2d:()=>ii,computeWorkgroupInfoForMatMul:()=>ni,computeWorkgroupSizeForConv2d:()=>ri,flatDispatchLayout:()=>V,isWebGPUSupported:()=>oi,tilesFitEvenlyIntoShape:()=>ti}),ei=e=>{let t=1;for(let n=0;n<e.length;n++)t*=e[n];return t};function ti(e,t){if(e.length!==t.length)throw Error(`Cannot compute whether rank ${e.length} tiles fit evenly into rank ${t.length} shape - ranks must match.`);return t.every((t,n)=>t%e[n]===0)}function B(e,t,n=[1,1,1],r=[1,1,1]){let[i,a,o]=[Math.ceil(ei(e.x.map(e=>t[e]))/(n[0]*r[0])),e.y?Math.ceil(ei(e.y.map(e=>t[e]))/(n[1]*r[1])):1,e.z?Math.ceil(ei(e.z.map(e=>t[e]))/(n[2]*r[2])):1];return[i,a,o]}function ni(e,t,n,r=!1){let i=[8,8,1],a=[4,4,1];return r||(e<=8&&(a[1]=1),t<=16&&n<=16&&(i[0]=4)),{workgroupSize:i,elementsPerThread:a}}function ri(e,t,n=!1){if(n)return[8,8,1];let r=ei(e.x.map(e=>t[e])),i=ei(e.y.map(e=>t[e]));return r<=4?[4,16,1]:i<=4?[16,4,1]:[16,16,1]}function ii(e,t,n=!1){if(n)return[4,4,1];let r=ei(e.x.map(e=>t[e])),i=ei(e.y.map(e=>t[e]));return r<=4?[1,2,1]:i<=4?[2,1,1]:[2,2,1]}function V(e){return{x:e.map((e,t)=>t)}}function ai(e){if(e===`float32`||e===`int32`||e===`bool`||e===`string`)return 4;if(e===`complex64`)return 8;throw Error(`Unknown dtype ${e}`)}function oi(){return!!(typeof globalThis<`u`&&globalThis.navigator&&globalThis.navigator.gpu)}function si(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{e!=null&&k(e.dtype!==`complex64`,()=>`${t} does not support complex64 tensors in the WebGPU backend.`)})}var H;(function(e){e[e.MatMulReduceProgram=0]=`MatMulReduceProgram`,e[e.MatMulSplitKProgram=1]=`MatMulSplitKProgram`,e[e.MatMulSmallOutputSizeProgram=2]=`MatMulSmallOutputSizeProgram`,e[e.MatMulPackedProgram=3]=`MatMulPackedProgram`,e[e.MatMulMax=4]=`MatMulMax`})(H||={});var ci=j().getNumber(`WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD`),li=(e,t)=>{let n=e.limits.maxComputeWorkgroupsPerDimension,r=t.dispatchLayout,i=t.dispatch;if(i.every(e=>e<=n))return i;k(i[0]>n&&r.y===void 0&&r.z===void 0,()=>`Dispatch size exceeds WebGPU limits in Y or Z dimension.`);let a=Math.ceil(Math.sqrt(i[0]));return a>n?(a=Math.ceil(Math.cbrt(i[0])),k(a<=n,()=>`Total dispatch size exceeds WebGPU maximum.`),[a,a,a]):[a,a,1]},ui=class e extends tn{nextDataId(){return e.nextDataId++}constructor(e,t){if(super(),this.commandQueueOwnedIds=new WeakSet,this.dispatchCountInPass=0,this.disposed=!1,this.downloadWaitMs=0,this.tensorDataPendingDisposal=[],this.queryResolveBuffer=null,this.querySet=null,this.querySetCount=2,this.stagingPendingDisposal=[],this.uniformPendingDisposal=[],this.uploadWaitMs=0,this.hasReadSyncWarned=!1,this.hasTimestampQueryWarned=!1,!oi())throw Error(`WebGPU is not supported on this device`);this.pipelineCache={},this.device=e,this.queue=e.queue,this.commandEncoder=null,this.computePassEncoder=null,this.adapterInfo=new Dr(t),this.supportTimestampQuery=this.device.features.has(`timestamp-query`),this.thresholdToIncreaseWorkgroups=this.adapterInfo.intelGPUGeneration>=12?16:8,this.bufferManager=new Or(this.device),this.textureManager=new Ar(this.device),this.tensorMap=new qt(this,It()),j().getBool(`WEBGPU_USE_PROFILE_TOOL`)&&(this.dummyCanvas=document.createElement(`canvas`),this.dummyCanvas.width=1,this.dummyCanvas.height=1,this.dummyContext=this.dummyCanvas.getContext(`webgpu`),this.dummyContext.configure({device:e,format:`bgra8unorm`}),document.body.appendChild(this.dummyCanvas))}floatPrecision(){return 32}disposeData(e,t=!1){if(!this.tensorMap.has(e))return!0;let n=this.tensorMap.get(e);return t?n.refCount=0:n.refCount--,n.refCount>0?!1:(n.complexTensorInfos!=null&&(this.disposeData(n.complexTensorInfos.real.dataId),this.disposeData(n.complexTensorInfos.imag.dataId)),this.commandQueueOwnedIds.has(e)?(this.tensorDataPendingDisposal.push(e),!0):(this.releaseResource(e),this.tensorMap.delete(e),!0))}memory(){return{numBytesInGPU:this.bufferManager.numBytesUsed,numBytesAllocatedInGPU:this.bufferManager.numBytesAllocated,unreliable:!1}}releaseResource(e){let t=this.tensorMap.get(e);t&&t.resource&&(t.external||(t.resource instanceof GPUBuffer?this.bufferManager.releaseBuffer(t.resource):t.resource instanceof GPUTexture&&this.textureManager.releaseTexture(t.resource)),t.resource=null)}refCount(e){return this.tensorMap.has(e)?this.tensorMap.get(e).refCount:0}incRef(e){let t=this.tensorMap.get(e);t.refCount++}decRef(e){if(this.tensorMap.has(e)){let t=this.tensorMap.get(e);t.refCount--}}write(e,t,n){if(n===`complex64`&&e!=null)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);let r={id:this.nextDataId()};return this.tensorMap.set(r,{dtype:n,shape:t,values:e,refCount:1}),r}move(e,t,n,r,i){if(r===`complex64`)throw Error(`Cannot write to a complex64 dtype. Please use tf.complex(real, imag).`);this.tensorMap.set(e,{dtype:r,shape:n,values:t,refCount:i})}submitQueue(){this.queue.submit([this.commandEncoder.finish()]),this.commandEncoder=null,this.dispatchCountInPass=0,this.commandQueueOwnedIds=new WeakSet,this.tensorDataPendingDisposal.forEach(e=>{this.releaseResource(e),this.tensorMap.delete(e)}),this.uniformPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e)),this.stagingPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e,!1)),this.tensorDataPendingDisposal=[],this.uniformPendingDisposal=[],this.stagingPendingDisposal=[]}ensureCommandEncoderReady(){this.commandEncoder||=this.device.createCommandEncoder()}endComputePassEncoder(){this.computePassEncoder&&=(this.computePassEncoder.end(),null)}async checkCompileCompletionAsync(){let e;try{e=await Promise.all(Object.values(this.pipelineCache))}catch(e){throw Error(e.message)}Object.keys(this.pipelineCache).map((t,n)=>{this.pipelineCache[t]=e[n]})}async getBufferData(e){if(j().getBool(`WEBGPU_ENGINE_COMPILE_ONLY`))return console.warn(`The data may be invalid since WEBGPU_ENGINE_COMPILE_ONLY is true, this can only be called when WEBGPU_ENGINE_COMPILE_ONLY is false`),null;let t=e.size,n=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,n,0,t),this.submitQueue(),await n.mapAsync(GPUMapMode.READ);let r=n.getMappedRange().slice(0);return n.unmap(),n!=null&&this.bufferManager.releaseBuffer(n),j().getBool(`WEBGPU_USE_PROFILE_TOOL`)&&(k(this.dummyContext!==void 0,()=>`Fail to get context for profiling tool`),this.dummyContext.getCurrentTexture()),r}convertAndCacheOnCPU(e,t){let n=this.tensorMap.get(e);return n.values=t,n.values}readSync(e){let t=this.tensorMap.get(e),{values:n,complexTensorInfos:r}=t;if(n!=null||t.dtype===`string`)return n;if(t.dtype===`complex64`){let t=this.readSync(r.real.dataId),n=this.readSync(r.imag.dataId),i=Cr(Ee(t,n).buffer,`float32`);return this.convertAndCacheOnCPU(e,i),i}this.hasReadSyncWarned||(this.hasReadSyncWarned=!0,console.warn(`The performance of synchronously reading data from GPU to CPU is poor on the webgpu backend, please use asynchronous APIs instead.`));let i=[`opaque`,`premultiplied`],a=t.resource,o=a.size;k(o%4==0,()=>`Because there is 4 bytes for one pixel, buffer size must be multiple of 4.`);let s=o/4,c=new ArrayBuffer(o),l=i.map(e=>new OffscreenCanvas(256,256)),u=new OffscreenCanvas(256,256);this.endComputePassEncoder(),l.map((e,t)=>{let n=e.getContext(`webgpu`);return n.configure({device:this.device,format:`bgra8unorm`,usage:GPUTextureUsage.COPY_DST,alphaMode:i[t]}),n.getCurrentTexture()}).map((e,t)=>{let n=(n,r,o)=>{this.ensureCommandEncoderReady(),this.commandEncoder.copyBufferToTexture({buffer:a,bytesPerRow:1024,offset:o},{texture:e},{width:n,height:r}),this.submitQueue();let s=u.getContext(`2d`,{willReadFrequently:!0});s.clearRect(0,0,n,r),s.drawImage(l[t],0,0);let d=s.getImageData(0,0,n,r).data,f=i[t],p=new Uint8ClampedArray(c,o,n*r*4);for(let e=0;e<p.length;e+=4)if(f===`premultiplied`)p[e+3]=d[e+3];else{let t=d[e];p[e]=d[e+2],p[e+1]=d[e+1],p[e+2]=t}},r=Math.floor(s/65536),o=256,d=256,f=0;for(let e=0;e<r;e++)n(o,d,f),f+=262144;let p=s%65536;d=Math.floor(p/256),d>0&&(n(o,d,f),f+=d*1024),o=p%256,o>0&&n(o,1,f)});let d=Cr(c,t.dtype);return this.convertAndCacheOnCPU(e,d),d}async read(e){if(!this.tensorMap.has(e))throw Error(`Tensor ${e} was not registered!`);let t=this.tensorMap.get(e),{values:n}=t;if(n!=null)return n;let r;if(t.dtype===`complex64`){let e=await Promise.all([this.read(t.complexTensorInfos.real.dataId),this.read(t.complexTensorInfos.imag.dataId)]),n=e[0],i=e[1];r=Ee(n,i)}else{let e=await this.getBufferData(t.resource);r=Cr(e,t.dtype)}return this.convertAndCacheOnCPU(e,r),r}copyBuffer(e){let t=e.size,n=e.usage,r=this.bufferManager.acquireBuffer(t,n);return this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,r,0,t),this.submitQueue(),r}createTensorFromGPUData(e,t,n){let r=e.buffer;if(n===`complex64`)throw Error(`Cannot write to a complex64 dtype. `);let i={id:this.nextDataId()};this.tensorMap.set(i,{dtype:n,shape:t,values:null,refCount:1,external:e.zeroCopy});let a=this.tensorMap.get(i),o=ai(a.dtype)*M(a.shape);if(e.buffer.size<o)throw Error(`GPUBuffer size(${e.buffer.size}) is smaller than tensor size(${o})!`);if((e.buffer.usage&(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))!==(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))throw Error(`GPUBuffer.usage should include GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC!`);return e.zeroCopy!==!0&&(r=this.copyBuffer(r)),a.resource=r,It().makeTensorFromDataId(i,t,n,this)}readToGPU(e){let{values:t,dtype:n,shape:r,resource:i}=this.tensorMap.get(e);if(n===`complex64`)throw Error(`Does not support reading buffer for complex64 dtype.`);if(i==null)throw Error(t==null?`There is no data on GPU or CPU.`:`Data is not on GPU but on CPU.`);let a=i,o=a.size,s=a.usage,c=this.bufferManager.acquireBuffer(o,s);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(i,0,c,0,o),this.submitQueue();let l=this.makeTensorInfo(r,n),u=It().makeTensorFromTensorInfo(l),d=this.tensorMap.get(l.dataId);return d.resource=c,{tensorRef:u,buffer:c}}bufferSync(e){let t=this.readSync(e.dataId);if(e.dtype===`string`)try{let n=t.map(e=>ye(e));return Oe(e.shape,e.dtype,n)}catch{throw Error(`Failed to decode encoded string bytes into utf-8`)}return Oe(e.shape,e.dtype,t)}async time(e){!this.supportTimestampQuery&&!this.hasTimestampQueryWarned&&(console.warn(`This device doesn't support timestamp-query extension. Start Chrome browser with flag --enable-dawn-features=allow_unsafe_apis to try it again. Otherwise, zero will be shown for the kernel time when profiling mode is enabled.`),this.hasTimestampQueryWarned=!0);let t=this.activeTimers,n=[],r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();let i=Ct(this.activeTimers.map(e=>e.query)).filter(e=>e!=null),a=Ct(this.activeTimers.map(e=>e.name)).filter(e=>e!=null);this.activeTimers=t,r&&(this.programTimersStack=null);let o={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null},s=await Promise.all(i);return o.kernelMs=Tn(s),o.getExtraProfileInfo=()=>s.map((e,t)=>({name:a[t],ms:e})).map(e=>`${e.name}: ${e.ms}`).join(`, `),this.uploadWaitMs=0,this.downloadWaitMs=0,o}makeTensorInfo(e,t,n){return t===`string`&&n!=null&&n.length>0&&_t(n[0])&&(n=n.map(e=>Hn(e))),{dataId:this.write(n,e,t),shape:e,dtype:t}}tensorToBinding(e){if(!e)return null;let t=this.tensorMap.get(e.dataId).resource;return t instanceof GPUBuffer?{buffer:t}:t instanceof GPUTexture?t.createView():t}uploadToGPU(e){let t=this.tensorMap.get(e);if(t.resource!=null)return;let n=ai(t.dtype)*M(t.shape),r,i=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;if(t.values){if(r=this.bufferManager.acquireBuffer(n,i,!0),r.mapState===`unmapped`){let e=this.bufferManager.acquireBuffer(n,GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,!0,!1),i=e.getMappedRange();t.dtype===`int32`||t.dtype===`bool`?new Int32Array(i).set(t.values):new Float32Array(i).set(t.values),e.unmap(),this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,r,0,n),this.stagingPendingDisposal.push(e)}else{let e=r.getMappedRange();t.dtype===`int32`||t.dtype===`bool`?new Int32Array(e).set(t.values):new Float32Array(e).set(t.values),r.unmap()}t.values=null}else r=this.bufferManager.acquireBuffer(n,i);t.resource=r}makeUniforms(e){let t=0,n=0,r=[],i=1;e.forEach(e=>{e.data.length===0&&(e.data=[1]);let a;switch(e.data.length){case 1:a=4;break;case 2:a=8;break;case 3:a=16;break;case 4:a=16;break;case 5:a=16;break;case 6:a=16;break;default:k(!1,()=>`Unsupported ${e.data.length}D shape`)}(n===5||n===6)&&(a=16),a>i&&(i=a),t=Math.ceil(t/a)*a,n=e.data.length,r.push(t),t+=e.data.length*4}),t=Math.ceil(t/i)*i;let a=new ArrayBuffer(t);e.forEach((e,t)=>{let n=r[t];e.type===`int32`?new Int32Array(a,n,e.data.length).set(e.data):e.type===`uint32`?new Uint32Array(a,n,e.data.length).set(e.data):new Float32Array(a,n,e.data.length).set(e.data)});let o=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.UNIFORM);return this.queue.writeBuffer(o,0,a,0,t),this.uniformPendingDisposal.push(o),{offset:0,size:t,buffer:o}}runWebGPUProgram(e,t,n,r,i){if(i||=this.makeTensorInfo(e.outputShape,n),M(i.shape)===0)return this.tensorMap.get(i.dataId).values=ct(i.dtype,0),i;this.uploadToGPU(i.dataId),e.dispatch=li(this.device,e);let a=t.map((t,n)=>{if(t.dtype===`complex64`)throw Error(`GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.`);return this.uploadToGPU(t.dataId),{dtype:this.tensorMap.get(t.dataId).dtype,shape:t.shape,name:e.variableNames[n]}});e.shaderKey=zr(e,a,i);let o=j().getBool(`WEBGPU_ENGINE_COMPILE_ONLY`);return e.shaderKey in this.pipelineCache||(this.pipelineCache[e.shaderKey]=Fr(this.device,e,a,i,o)),e.pipeline=this.pipelineCache[e.shaderKey],o||this.recordAndSubmit(e,i,t,r),i}recordAndSubmit(e,t,n,r){if(e.pipeline instanceof Promise)throw Error(`Please call checkCompileCompletionAsync to ensure parallel compilation is done!`);let i=[],a=[],o=`int32`;if(e.pixelsOpType==null){i.push({type:`float32`,data:[NaN]},{type:`float32`,data:[1/0]}),a=n.concat(t).map(e=>e.shape);let e=`int32`;a.map(t=>{i.push({type:e,data:t});let n=We(t);i.push({type:e,data:n})})}else{let e=We(t.shape);i.push({type:o,data:e})}if(e.size){let t=M(e.outputShape);i.push({type:o,data:[e.outputComponent?t/e.outputComponent:t]})}r&&(i=[...i,...r]);let s=[this.tensorToBinding(t),...n.map(e=>this.tensorToBinding(e)),this.makeUniforms(i)];n.forEach(e=>{this.commandQueueOwnedIds.add(e.dataId)}),this.commandQueueOwnedIds.add(t.dataId);let c=this.device.createBindGroup({layout:e.pipeline.getBindGroupLayout(0),entries:s.map((e,t)=>({binding:t,resource:e}))}),l=this.activeTimers!=null;this.ensureCommandEncoderReady();let u={};l&&this.supportTimestampQuery?(this.endComputePassEncoder(),this.querySet??=this.device.createQuerySet({type:`timestamp`,count:this.querySetCount}),u.timestampWrites={querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1},this.computePassEncoder=this.commandEncoder.beginComputePass(u)):this.computePassEncoder||=this.commandEncoder.beginComputePass(u),this.computePassEncoder.setPipeline(e.pipeline),this.computePassEncoder.setBindGroup(0,c),this.computePassEncoder.dispatchWorkgroups(e.dispatch[0],e.dispatch[1],e.dispatch[2]),this.dispatchCountInPass++,(l||j().get(`WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE`)<=this.dispatchCountInPass||e.pixelsOpType===Pr.DRAW)&&(this.endComputePassEncoder(),l?this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime()}):this.submitQueue())}async getQueryTime(){if(!this.supportTimestampQuery)return 0;this.queryResolveBuffer??=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.QUERY_RESOLVE),this.commandEncoder.resolveQuerySet(this.querySet,0,this.querySetCount,this.queryResolveBuffer,0);let e=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer,0,e,0,this.querySetCount*8),this.submitQueue(),await e.mapAsync(GPUMapMode.READ);let t=new BigUint64Array(e.getMappedRange()),n=Number(t[1]-t[0])/1e6;return e.unmap(),this.bufferManager.releaseBuffer(e),n}shouldExecuteOnCPU(e,t=ci){return j().getBool(`WEBGPU_CPU_FORWARD`)&&e.every(e=>this.tensorMap.get(e.dataId).resource==null&&M(e.shape)<t)}numDataIds(){return this.tensorMap.numDataIds()-this.tensorDataPendingDisposal.length}dispose(){this.disposed||=(this.querySet!=null&&this.querySet.destroy(),this.bufferManager.dispose(),this.textureManager.dispose(),!0)}};ui.nextDataId=0,oi()&&Rn(`webgpu`,async()=>{let e={powerPreference:j().get(`WEBGPU_USE_LOW_POWER_GPU`)?`low-power`:`high-performance`},t=await navigator.gpu.requestAdapter(e),n={},r=[];t.features.has(`timestamp-query`)&&r.push(`timestamp-query`),t.features.has(`bgra8unorm-storage`)&&r.push([`bgra8unorm-storage`]),n.requiredFeatures=r;let i=t.limits;return n.requiredLimits={maxComputeWorkgroupStorageSize:i.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:i.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:i.maxStorageBufferBindingSize,maxBufferSize:i.maxBufferSize,maxComputeWorkgroupSizeX:i.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:i.maxComputeInvocationsPerWorkgroup},new ui(await t.requestDevice(n),`info`in t?t.info:`requestAdapterInfo`in t?await t.requestAdapterInfo():void 0)},3);var U;(function(e){e[e.ADD=0]=`ADD`,e[e.ATAN2=1]=`ATAN2`,e[e.COMPLEX_MULTIPLY_IMAG=2]=`COMPLEX_MULTIPLY_IMAG`,e[e.COMPLEX_MULTIPLY_REAL=3]=`COMPLEX_MULTIPLY_REAL`,e[e.DIV=4]=`DIV`,e[e.ELU_DER=5]=`ELU_DER`,e[e.EQUAL=6]=`EQUAL`,e[e.FLOOR_DIV=7]=`FLOOR_DIV`,e[e.GREATER=8]=`GREATER`,e[e.GREATER_EQUAL=9]=`GREATER_EQUAL`,e[e.LESS=10]=`LESS`,e[e.LESS_EQUAL=11]=`LESS_EQUAL`,e[e.LOGICAL_AND=12]=`LOGICAL_AND`,e[e.LOGICAL_OR=13]=`LOGICAL_OR`,e[e.MAX=14]=`MAX`,e[e.MIN=15]=`MIN`,e[e.MOD=16]=`MOD`,e[e.MUL=17]=`MUL`,e[e.NOT_EQUAL=18]=`NOT_EQUAL`,e[e.POW=19]=`POW`,e[e.PRELU=20]=`PRELU`,e[e.SQUARED_DIFFERENCE=21]=`SQUARED_DIFFERENCE`,e[e.SUB=22]=`SUB`})(U||={});var di=`let resultTemp = a + b;`,fi=`let resultTemp = atan2(a, b);`,pi=`let resultTemp = areal * breal - aimag * bimag;`,mi=`let resultTemp = areal * bimag + aimag * breal;`,hi=`let resultTemp = a / b;`,gi=`let resultTemp = select(a * (b + 1.0), a, b >= b - b);`,_i=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a == b);
`,vi=`
  let remainder =
      select(a % b, round(a % b), (round(a) == a) & (round(b) == b));
  let quotient = (a - remainder) / b;
  let resultTemp =
      round(select(quotient, quotient - 1, sign(remainder) == -sign(b)));
`,yi=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a > b);
`,bi=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a >= b);
`,xi=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a < b);
`,Si=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a <= b);
`,Ci=`return f32(a >= 1.0 && b >= 1.0);`,wi=`return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`,Ti=`return f32(a >= 1.0 || b >= 1.0);`,Ei=`return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`,Di=`let resultTemp = max(a, b);`,Oi=`let resultTemp = min(a, b);`,ki=`
  let isNaN = b == 0.;
  var resultTemp = a % b;
  resultTemp = select((resultTemp + b) % b, resultTemp,
      (a < 0. && b < 0.) || (a >= 0. && b > 0.));
`,Ai=`
  let isNaN = !vec4<bool>(b);
  var resultTemp = vec4<f32>(a % b);
  if (!((a[0] < 0. && b[0] < 0.) || (a[0] >= 0. && b[0] > 0.))) {
    resultTemp[0] = (resultTemp[0] + b[0]) % b[0];
  }
  if (!((a[1] < 0. && b[1] < 0.) || (a[1] >= 0. && b[1] > 0.))) {
    resultTemp[1] = (resultTemp[1] + b[1]) % b[1];
  }
  if (!((a[2] < 0. && b[2] < 0.) || (a[2] >= 0. && b[2] > 0.))) {
    resultTemp[2] = (resultTemp[2] + b[2]) % b[2];
  }
  if (!((a[3] < 0. && b[3] < 0.) || (a[3] >= 0. && b[3] > 0.))) {
    resultTemp[3] = (resultTemp[3] + b[3]) % b[3];
  }
`,ji=`let resultTemp = a * b;`,Mi=`
  var resultTemp = f32(a != b);
  let valueForNaN = 1.0;
`,Ni=`
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
`,Pi=`
  let isNaN = a < 0.0 && floor(b) < b;
  if (b == 0.0) {
    return 1.0;
  }
  var resultTemp = select(sign(a) * pow(abs(a), b), pow(abs(a), b),
      round(abs(b) % 2.0) != 1.0);
`,Fi=`
  let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = (a < vec4<f32>(0.0)) & (floor(b) < b);
`,Ii=`if (a < 0.0) { return b * a; }  return a;`,Li=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,Ri=`let resultTemp = (a - b) * (a - b);`,zi=`let resultTemp = a - b;`;function Bi(e,t){let n;do{switch(e){case U.ATAN2:n=fi;break;case U.MAX:n=Di;break;case U.MIN:n=Oi;break;case U.MOD:n=t?Ai:ki;break;case U.NOT_EQUAL:n=t?Ni:Mi;break;case U.POW:n=t?Fi:Pi;break;default:continue}let r,i,a;return t?(r=`isnanVec4`,i=`vec4<f32>`,a=`vec4<bool>`):(r=`isnan`,i=`f32`,a=`bool`),`
      let aIsNaN = ${r}(a);
      let aPostLegalization = select(a, ${i}(42), aIsNaN);
      let bIsNaN = ${r}(b);
      let bPostLegalization = select(b, ${i}(42), bIsNaN);
      let isNaN = false;
      let valueForNaN = uniforms.NAN;
      {
        let a = aPostLegalization;
        let b = bPostLegalization;
        ${n}
        return select(
            resultTemp, ${i}(valueForNaN),
            ${a}(isNaN) | aIsNaN | bIsNaN);
      }
    `}while(0);switch(e){case U.ADD:n=di;break;case U.COMPLEX_MULTIPLY_IMAG:n=mi;break;case U.COMPLEX_MULTIPLY_REAL:n=pi;break;case U.DIV:n=hi;break;case U.ELU_DER:n=gi;break;case U.EQUAL:n=_i;break;case U.FLOOR_DIV:n=vi;break;case U.GREATER:n=yi;break;case U.GREATER_EQUAL:n=bi;break;case U.LESS:n=xi;break;case U.LESS_EQUAL:n=Si;break;case U.LOGICAL_AND:return t?wi:Ci;case U.LOGICAL_OR:return t?Ei:Ti;case U.MUL:n=ji;break;case U.PRELU:return t?Li:Ii;case U.SQUARED_DIFFERENCE:n=Ri;break;case U.SUB:n=zi}return`
    ${n}
    return resultTemp;
  `}var W;(function(e){e[e.ABS=0]=`ABS`,e[e.ACOS=1]=`ACOS`,e[e.ACOSH=2]=`ACOSH`,e[e.ASIN=3]=`ASIN`,e[e.ASINH=4]=`ASINH`,e[e.ATAN=5]=`ATAN`,e[e.ATANH=6]=`ATANH`,e[e.CEIL=7]=`CEIL`,e[e.COS=8]=`COS`,e[e.COSH=9]=`COSH`,e[e.ELU=10]=`ELU`,e[e.ERF=11]=`ERF`,e[e.EXP=12]=`EXP`,e[e.EXPM1=13]=`EXPM1`,e[e.FLOOR=14]=`FLOOR`,e[e.IS_FINITE=15]=`IS_FINITE`,e[e.IS_INF=16]=`IS_INF`,e[e.IS_NAN=17]=`IS_NAN`,e[e.LINEAR=18]=`LINEAR`,e[e.LOG=19]=`LOG`,e[e.LOG1P=20]=`LOG1P`,e[e.LOGICAL_NOT=21]=`LOGICAL_NOT`,e[e.NEG=22]=`NEG`,e[e.RELU=23]=`RELU`,e[e.RELU6=24]=`RELU6`,e[e.LEAKYRELU=25]=`LEAKYRELU`,e[e.RECIPROCAL=26]=`RECIPROCAL`,e[e.ROUND=27]=`ROUND`,e[e.RSQRT=28]=`RSQRT`,e[e.SELU=29]=`SELU`,e[e.SIGMOID=30]=`SIGMOID`,e[e.SIGN=31]=`SIGN`,e[e.SIN=32]=`SIN`,e[e.SINH=33]=`SINH`,e[e.SOFTPLUS=34]=`SOFTPLUS`,e[e.SQRT=35]=`SQRT`,e[e.SQUARE=36]=`SQUARE`,e[e.STEP=37]=`STEP`,e[e.TAN=38]=`TAN`,e[e.TANH=39]=`TANH`,e[e.TO_INT=40]=`TO_INT`})(W||={});var Vi=`return abs(a);`,Hi=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return acos(a);
`,Ui=`
  if (a < 1.) {
    return uniforms.NAN;
  }
  return acosh(a);
`,Wi=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return asin(a);
`,Gi=`return asinh(a);`,Ki=`
  if (isnan(a)) {
    return uniforms.NAN;
  }
  return atan(a);
`,qi=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  if (a == 1.) {
    return uniforms.INFINITY;
  }
  if (a == -1.) {
    return -uniforms.INFINITY;
  }
  return atanh(a);
`,Ji=`return ceil(a);`,Yi=`return cos(a);`,Xi=`
  let e2x = exp(-a);
  return (e2x + 1.0 / e2x) / 2.0;
`,Zi=`return exp(a) - 1.0;`,Qi=`if (a >= 0.0) { return a; }  return (exp(a) - 1.0);`,$i=`
  var resFloat = exp(a) - vec4<f32>(1.0);
  if (a.r >= 0.0) {
    resFloat.r = a.r;
  }
  if (a.g >= 0.0) {
    resFloat.g = a.g;
  }
  if (a.b >= 0.0) {
    resFloat.b = a.b;
  }
  if (a.a >= 0.0) {
    resFloat.a = a.a;
  }
  return resFloat;
`,ea=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  let p = ${ht};
  let a1 = ${Er};
  let a2 = ${h};
  let a3 = ${ut};
  let a4 = ${de};
  let a5 = ${ot};

  let sign = sign(a);
  let absA = abs(a);
  let t = 1.0 / (1.0 + p * absA);
  return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absA * absA));
`,ta=`return exp(a);`,na=`return floor(a);`,ra=`return f32(!isnan(a) && !isinf(a));`,ia=`return f32(isinf(a));`,aa=`return f32(isnan(a));`,oa=`return a;`,sa=`if (a < 0.0) { return uniforms.NAN; }
  return log(a);`,ca=`
  if (isnan(a)) { return a; }
  return log(1.0 + a);
`,la=`return f32(!(a >= 1.0));`,ua=`return -a;`,da=`if (a < 0.0) { return uniforms.alpha * a; } return a;`,fa=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (uniforms.alpha * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,pa=`return 1.0 / a;`,ma=`return select(a, 0.0, a < 0.0);`,ha=`return clamp(a, 0.0, 6.0);`,ga=`return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));`,_a=`
  return select(a, vec4<f32>(0.0), a < vec4<f32>(0.0));
`,va=`return round(a);`,ya=`return inverseSqrt(a);`,ba=`
  if (a >= 0.0) {
    return ${oe} * a;
  } else {
    return ${xe} * (exp(a) - 1.0);
  }
`,xa=`return 1.0 / (1.0 + exp(-1.0 * a));`,Sa=`return sign(a);`,Ca=`return sin(a);`,wa=`
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`,Ta=`
  let epsilon = 1.1920928955078125e-7;
  let threshold = log(epsilon) + 2.0;

  let too_large = a > -threshold;
  let too_small = a < threshold;
  let exp_a = exp(a);

  if (too_large) {
    return a;
  } else if (too_small) {
    return exp_a;
  } else {
    return log(exp_a + 1.0);
  }
`,Ea=`return sqrt(a);`,Da=`return a * a;`,Oa=`
  if (isnan(a)) {
    return a;
  }

  return select(uniforms.stepAlpha, 1.0, a > 0.0);
`,ka=`return tan(a);`,Aa=`
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`,ja=`return f32(i32((a)));`;function G(e,t){switch(e){case W.ABS:return Vi;case W.ACOS:return Hi;case W.ACOSH:return Ui;case W.ASIN:return Wi;case W.ASINH:return Gi;case W.ATAN:return Ki;case W.ATANH:return qi;case W.COS:return Yi;case W.COSH:return Xi;case W.CEIL:return Ji;case W.ELU:return t?$i:Qi;case W.ERF:return ea;case W.EXP:return ta;case W.EXPM1:return Zi;case W.FLOOR:return na;case W.IS_FINITE:return ra;case W.IS_INF:return ia;case W.IS_NAN:return aa;case W.LINEAR:return oa;case W.LOG:return sa;case W.LOG1P:return ca;case W.LOGICAL_NOT:return la;case W.NEG:return ua;case W.LEAKYRELU:return t?fa:da;case W.RECIPROCAL:return pa;case W.RELU:return t?_a:ma;case W.RELU6:return t?ga:ha;case W.ROUND:return va;case W.RSQRT:return ya;case W.SELU:return ba;case W.SIGMOID:return xa;case W.SIGN:return Sa;case W.SIN:return Ca;case W.SINH:return wa;case W.SOFTPLUS:return Ta;case W.SQRT:return Ea;case W.SQUARE:return Da;case W.STEP:return Oa;case W.TAN:return ka;case W.TANH:return Aa;case W.TO_INT:return ja;default:throw Error(`BinaryType ${e} is not implemented!`)}}function K(e,t=!1,n=!1,r=3){if(e===null)return``;let i=``;if(e===`linear`)i=G(W.LINEAR);else if(e===`relu`)i=G(W.RELU,n);else if(e===`elu`)i=G(W.ELU,n);else if(e===`relu6`)i=G(W.RELU6,n);else if(e===`prelu`)i=Bi(U.PRELU,n);else if(e===`sigmoid`)i=G(W.SIGMOID,n);else if(e===`leakyrelu`)i=G(W.LEAKYRELU,n);else throw Error(`Activation ${e} has not been implemented for the WebGPU backend.`);let a=I(n?4:1),o=``;return o=t?`
      fn activation(a : ${a}, coords : vec${r}<i32>) -> ${a} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${i}
      }`:`
      fn activation(a : ${a}, coords : vec${r}<i32>) -> ${a} {
        ${i}
      }`,o}function Ma(e,t){return`
      ${e?`value = value + getBiasByOutputCoords(coords);`:``}
      ${t?`value = activation(value, coords);`:``}
      `}function Na(e,t,n=!1,r=!1,i=!1,a=1){k(e&&a===1||!e,()=>`transposeA ${e} is not compatible with component size ${a}`);let o=`
      ${e?`value = getA(batch, col, row);`:`value = getA(batch, row, col);`}

    `,s=t?`value = getB(batch, col, row);`:`value = getB(batch, row, col);`;return`
  fn mm_readA(batch: i32, row: i32, col: i32) -> ${I(a)} {
    var value = ${I(a)}(0.0);
    ${n&&i?o:`
    ${e?`if(row < uniforms.dimAOuter && col < uniforms.dimInner)`:`if(row < uniforms.aShape[1] && col < uniforms.aShape[2])`}
    {
      ${o}
    }
    `}
    return value;
  }

  fn mm_readB(batch: i32, row: i32, col: i32) -> ${I(a)} {
    var value = ${I(a)}(0.0);
    ${s}
    return value;
  }
  `}function Pa(e,t,n,r,i=!1,a=!1,o=!1,s=1){return`
  ${Na(n,r,i,a,o,s)}
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: ${I(s)}) {
    ${i&&a?``:`if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)`}
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      ${Ma(e,t)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  `}var Fa=(e,t)=>e?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol * ${t});
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * ${t});
        `,Ia=(e,t,n,r)=>{if(e)return`
      for (var k = 0; k < ${r}; k++) {
        let BCached0 = mm_Bsub[k][tileCol];
        let ACached0 = mm_Asub[k][localRow];
        for (var i = 0; i < ${n}; i++) {
          acc[i] = fma(BCached0, vec4<f32>(ACached0[i]), acc[i]);
        }
      }`;{let e=``,i=``;for(let n=0;n<t;n++)e+=`let BCached${n} = mm_Bsub[k * ${t} + ${n}][tileCol];`,i+=`acc[i] = fma(BCached${n}, vec4<f32>(ACached[${n}]), acc[i]);`;return`
      for (var k = 0; k < ${r/t}; k++) {
        ${e}
        for (var i = 0; i < ${n}; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          ${i}
        }
      }`}};function La(e,t,n=!1,r=32,i=!1,a=32,o=!1){let s=t[1]*e[1],c=t[0]*e[0],l=n?s:r,u=n?r:s,d=l/t[0],f=r/t[1],p=e[1],m=e[0];return k((n&&d===4&&e[1]===4||!n&&(d===3||d===4))&&l%t[0]===0&&r%t[1]===0&&e[0]===4,()=>`If transposeA ${n} is true, innerElementSize ${d} and workPerThread[1] ${e[1]} must be 4.
          Otherwise, innerElementSize ${d} must be 3 or 4.
      tileAWidth ${l} must be divisible by workgroupSize[0]${t[0]}. tileInner ${r} must be divisible by workgroupSize[1] ${t[1]}. colPerThread ${e[0]} must be 4.`),`
  var<workgroup> mm_Asub : array<array<vec${d}<f32>, ${l/d}>, ${u}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${c/e[0]}>, ${r}>;

  ${z()} {
    let localRow = i32(localId.y);
    let tileRow = localRow * ${p};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * ${p};
    let globalCol = i32(globalId.x) * ${m};
    let batch = ${i?`0`:`i32(globalId.z)`};
    let batchA = ${i||!o?`batch`:`batch % uniforms.aShape[0]`};
    let batchB = ${i||!o?`batch`:`batch % uniforms.bShape[0]`};
    let globalRowStart = i32(workgroupId.y) * ${s};

    let numTiles = ${i?`${Math.ceil(a/r)}`:`(uniforms.dimInner - 1) / ${r} + 1`};
    var kStart = ${i?`i32(globalId.z) * ${a}`:`0`};

    var acc: array<vec4<f32>, ${p}>;

    // Loop over shared dimension.
    let tileRowB = localRow * ${f};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${p}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${Fa(n,d)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${f}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + ${r};
        workgroupBarrier();

        // Compute acc values for a single thread.
        ${Ia(n,d,p,r)}
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < ${p}; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`}var Ra=e=>e?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol);
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRowStart + inputRow,
          kStart + inputCol);
        `,za=e=>e?`let ACached = mm_Asub[k][tileRow + innerRow];`:`let ACached = mm_Asub[tileRow + innerRow][k];`;function Ba(e,t,n=!1,r=32,i=!1,a=32,o=!1,s=!1){let c=e[1]*t[1],l=e[0]*t[0],u=n?c:r,d=n?r:c;k(d%t[1]===0&&u%t[0]===0&&r%t[1]===0,()=>`tileAHight ${d} must be divisible by workgroupSize[1]${t[1]}, tileAWidth ${u} must be divisible by workgroupSize[0]${t[0]}, tileInner ${r} must be divisible by workgroupSize[1]${t[1]}`);let f=d/t[1],p=u/t[0],m=r/t[1],h=e[1],g=e[0],_=o?`
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${c};
      let globalColStart = i32(workgroupId.x) * ${l};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var inputRow = localRow; inputRow < ${d}; inputRow = inputRow + ${t[1]}) {
          for (var inputCol = localCol; inputCol < ${u}; inputCol = inputCol + ${t[0]}) {
            ${Ra(n)}
          }
        }
        // Load one tile of B into local memory.
        for (var inputRow = localRow; inputRow < ${r}; inputRow = inputRow + ${t[1]}) {
              for (var inputCol = localCol; inputCol < ${l}; inputCol = inputCol + ${t[0]}) {
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
              kStart + inputRow,
              globalColStart + inputCol);
          }
        }
        kStart = kStart + ${r};
        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ${g}>;
        for (var k = 0; k < ${r}; k++) {
          for (var inner = 0; inner < ${g}; inner++) {
            BCached[inner] = mm_Bsub[k][localCol + inner * ${t[0]}];
          }
          for (var innerRow = 0; innerRow < ${h}; innerRow++) {
            let ACached = ${n?`mm_Asub[k][localRow + innerRow * ${t[1]}];`:`mm_Asub[localRow + innerRow * ${t[1]}][k];`}
            for (var innerCol = 0; innerCol < ${g}; innerCol++) {
              acc[innerRow][innerCol] =
                  fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < ${h}; innerRow++) {
        let gRow = globalRowStart + localRow + innerRow * ${t[1]};
        for (var innerCol = 0; innerCol < ${g}; innerCol++) {
          let gCol = globalColStart + localCol + innerCol * ${t[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      `:`
  let tileRow = i32(localId.y) * ${h};
  let tileCol = i32(localId.x) * ${g};

  let globalRow = i32(globalId.y) * ${h};
  let globalCol = i32(globalId.x) * ${g};
  let globalRowStart = i32(workgroupId.y) * ${c};

  let tileRowA = i32(localId.y) * ${f};
  let tileColA = i32(localId.x) * ${p};
  let tileRowB = i32(localId.y) * ${m};
  // Loop over shared dimension.
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < ${f}; innerRow++) {
      for (var innerCol = 0; innerCol < ${p}; innerCol++) {
        let inputRow = tileRowA + innerRow;
        let inputCol = tileColA + innerCol;
        ${Ra(n)}
      }
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < ${m}; innerRow++) {
      for (var innerCol = 0; innerCol < ${g}; innerCol++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol + innerCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
          kStart + inputRow,
          globalCol + innerCol);
      }
    }
    kStart = kStart + ${r};
    workgroupBarrier();

    // Compute acc values for a single thread.
    var BCached : array<f32, ${g}>;
    for (var k = 0; k < ${r}; k++) {
      for (var inner = 0; inner < ${g}; inner++) {
        BCached[inner] = mm_Bsub[k][tileCol + inner];
      }

      for (var innerRow = 0; innerRow < ${h}; innerRow++) {
        ${za(n)}
        for (var innerCol = 0; innerCol < ${g}; innerCol++) {
          acc[innerRow][innerCol] =
              fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < ${h}; innerRow++) {
    for (var innerCol = 0; innerCol < ${g}; innerCol++) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;return`
    var<workgroup> mm_Asub : array<array<f32, ${u}>, ${d}>;
    var<workgroup> mm_Bsub : array<array<f32, ${l}>, ${r}>;

    ${z()} {
      let batch = ${i?`0`:`i32(globalId.z)`};
      let batchA = ${i||!s?`batch`:`batch % uniforms.aShape[0]`};
      let batchB = ${i||!s?`batch`:`batch % uniforms.bShape[0]`};
      let numTiles = ${i?`${Math.ceil(a/r)}`:`(uniforms.dimInner - 1) / ${r} + 1`};
      var kStart = ${i?`i32(globalId.z) * ${a}`:`0`};

      var acc : array<array<f32, ${g}>, ${h}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${h}; innerRow++) {
        for (var innerCol = 0; innerCol < ${g}; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${_}
    }
  `}var Va=e=>e?`
      mm_readA(batchA, colA, globalRow),
      mm_readA(batchA, colA + 1, globalRow),
      mm_readA(batchA, colA + 2, globalRow),
      mm_readA(batchA, colA + 3, globalRow)
  `:`
      mm_readA(batchA, globalRow, colA),
      mm_readA(batchA, globalRow, colA + 1),
      mm_readA(batchA, globalRow, colA + 2),
      mm_readA(batchA, globalRow, colA + 3)
  `;function Ha(e,t=!1){k(e[1]===1&&e[2]===1,()=>`A linear work group size is required. But got ${e}.`);let n=e[0]*4;return`
    var<workgroup> mm_Asub : array<vec4<f32>, ${e[0]}>;

    ${z()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / ${n} + 1;
      let batch = i32(globalId.z);
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * ${n} + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${Va(t)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < ${n/4}; k++) {
          let rowB = t * ${n} + k * 4;
          let BCached = vec4<f32>(mm_readB(batchB, rowB, globalCol),
                              mm_readB(batchB, rowB + 1, globalCol),
                              mm_readB(batchB, rowB + 2, globalCol),
                              mm_readB(batchB, rowB + 3, globalCol));

          let ACached = mm_Asub[k];
          acc = acc + dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      mm_write(batch, globalRow, globalCol, acc);
    }
  `}var Ua=class{constructor(e,t,n=!1,r=!1,i=null,a=null,o=null,s=!1){this.variableNames=[`A`,`B`],this.uniforms=`dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.outputShape=t,this.dispatchLayout={x:[2],y:[1],z:[0]};let c=n?e[1]:e[2];if(this.isVec4=(c%4==0&&!n||t[1]%4==0&&n)&&t[2]%4==0&&!r,this.outputComponent=this.isVec4?4:1,this.isVectorA=t[1]===1&&!n,!this.isVec4&&this.isVectorA)this.elementsPerThread=[1,1,1],this.workgroupSize=[32,1,1];else{let e=ni(t[1],c,t[2],n);this.workgroupSize=e.workgroupSize,this.elementsPerThread=e.elementsPerThread}this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread);let l=i!=null,u=o!=null;l&&this.variableNames.push(`bias`),u&&this.variableNames.push(`preluActivationWeights`),this.sequentialAccessByThreads=s,this.transposeA=n,this.transposeB=r,this.addBias=l,this.activation=a,this.hasPreluActivationWeights=u,[this.fitAOuter,this.fitBOuter,this.fitInner]=this.getShapeFit(t[1],t[2],c),this.shaderKey=`matMulPacked_${this.elementsPerThread}_${n}_${r}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.isVectorA}_${this.sequentialAccessByThreads}`}getShapeFit(e,t,n){let r=this.workgroupSize[1]*this.elementsPerThread[1],i=this.workgroupSize[0]*this.elementsPerThread[0];return this.tileInner=!this.isVec4&&this.isVectorA?this.workgroupSize[0]*4:i,[e%r===0,t%i===0,n%this.tileInner===0]}getUserCode(){return`
      ${K(this.activation,this.hasPreluActivationWeights,this.isVec4)}
      ${Pa(this.addBias,this.activation,!1,this.transposeB,this.fitAOuter,this.fitBOuter,this.fitInner,this.isVec4?4:1)}
      ${this.isVec4?La(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,!0):this.isVectorA?Ha(this.workgroupSize,this.transposeA):Ba(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,this.sequentialAccessByThreads,!0)}
    `}};function Wa(e){return`
    var<workgroup> sumValues : array<f32, ${e}>;
    ${z()} {
      let coords = getOutputCoords();
      let batch = coords[0];
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      let row = coords[1];
      let col = coords[2];
      var sum = 0.0;
      let Length = uniforms.dimInner;
      for (var k = i32(localId.x); k < Length; k = k + ${e}) {
        let dataA = mm_readA(batchA, row, k);
        let dataB = mm_readB(batchB, k, col);
        sum = sum + dataA * dataB;
      }
      sumValues[localId.x] = sum;
      workgroupBarrier();

      for(var currentSize = ${e/2}u; currentSize > 1u;
          currentSize = currentSize / 2u) {
        if (localId.x < currentSize)
        {
          sumValues[localId.x] = sumValues[localId.x] + sumValues[localId.x + currentSize];
        }
        workgroupBarrier();
      }

      if (localId.x == 0u) {
        sum = sumValues[0] + sumValues[1];
        mm_write(batch, row, col, sum);
      }
    }
  `}var Ga=class{constructor(e,t=!1,n=!1,r=null,i=null,a=null){this.variableNames=[`A`,`B`],this.uniforms=`dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.workgroupSize=[256,1,1],this.outputShape=e,this.dispatchLayout={x:[],y:[1,2],z:[0]},this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize);let o=r!=null,s=a!=null;o&&this.variableNames.push(`bias`),s&&this.variableNames.push(`preluActivationWeights`),this.transposeA=t,this.transposeB=n,this.addBias=o,this.activation=i,this.hasPreluActivationWeights=s,this.shaderKey=`matMulReduce_${this.activation}_${t}_${n}`}getUserCode(){return`
      ${K(this.activation,this.hasPreluActivationWeights)}
      ${Pa(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${Wa(this.workgroupSize[0])}
    `}};function Ka(e){let t=e[1],n=e[0],r=t>n?t:n;return`
  var<workgroup> mm_Asub : array<array<f32, ${r}>, ${t}>;
  var<workgroup> mm_Bsub : array<array<f32, ${n}>, ${r}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Read data from global memory to registers firstly, then store them into
  // shared memory, so it is instruction-Level parallelism for arithmetic
  // operations and others handle IO operations between barrier api, makes ALU
  // and load/store units work simultaneously, could improves the performance.
  ${z()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${r} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = 0;
    var regA = mm_readA(batchA, globalRow, globalColA);
    var regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
    var regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
    globalColA = globalColA + ${r};
    globalRowB = globalRowB + ${r};

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = regA;
      mm_Bsub[2 * tileRow][tileCol] = regB0;
      mm_Bsub[2 * tileRow + 1][tileCol] = regB1;

      workgroupBarrier();

      regA = mm_readA(batchA, globalRow, globalColA);
      regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
      regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
      globalColA = globalColA + ${r};
      globalRowB = globalRowB + ${r};

      for (var k = 0; k < ${r}; k = k + 1) {
        acc = acc + mm_Asub[tileRow][k] * mm_Bsub[k][tileCol];
      }
      workgroupBarrier();
    }

    mm_write(batch, globalRow, globalCol, acc);
  }
  `}var qa=class{constructor(e,t,n,r=!1,i=!1,a=null,o=null,s=null){this.variableNames=[`A`,`B`],this.uniforms=`dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.workgroupSize=[16,8,1],this.outputShape=n,this.dispatchLayout={x:[2],y:[1],z:[0]},this.dispatch=[Math.ceil(n[2]/this.workgroupSize[0]),Math.ceil(n[1]/this.workgroupSize[1]),n[0]];let c=a!=null;c&&this.variableNames.push(`bias`);let l=s!=null;l&&this.variableNames.push(`preluActivationWeights`),this.transposeA=r,this.transposeB=i,this.addBias=c,this.activation=o,this.hasPreluActivationWeights=l,this.shaderKey=`matMulSmallOutputSize_${this.activation}_${r}_${i}`}getUserCode(){return`
      ${K(this.activation,this.hasPreluActivationWeights)}
      ${Pa(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${Ka(this.workgroupSize)}
    `}},Ja=class{constructor(e,t,n=!1,r=!1){this.variableNames=[`A`,`B`],this.uniforms=`dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.workgroupSize=[8,8,1],this.atomic=!0,this.splitedDimInner=128,k(e[0]===1,()=>`MatMulSplitKProgram only supports batch = 1.`),this.outputShape=e,this.dispatchLayout={x:[2],y:[1],z:[0,3]};let i=(n&&this.outputShape[1]%4==0||!n&&t%4==0)&&this.outputShape[2]%4==0;this.elementsPerThread=[4,4,this.splitedDimInner],this.outputComponent=i?4:1,i||(this.outputShape[1]<16&&(this.elementsPerThread[1]=1),this.outputShape[2]<16&&(this.elementsPerThread[0]=1)),this.dispatch=B(this.dispatchLayout,[this.outputShape[0],this.outputShape[1],this.outputShape[2],t],this.workgroupSize,this.elementsPerThread),this.transposeA=n,this.transposeB=r,this.shaderKey=`matMulSplitK_${n}_${r}_${this.elementsPerThread}_${this.outputComponent}`}getUserCode(){let e=this.outputComponent;return`
      ${Na(!1,this.transposeB,!1,!1,!1,e)}
      fn mm_write(batch: i32, row : i32, col : i32, value : ${I(e)}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          for (var i = 0; i < ${e}; i = i + 1) {
            ${F(`&result[flatIndex + i]`,`${e>1?`value[i]`:`value`}`,`float32`)}
          }
        }
      }
      ${e===4?La(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner):Ba(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner)}
    `}},Ya=class{constructor(e,t=null,n=null,r=null){this.uniforms=``,this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t!=null,this.hasPreluActivationWeights=r!=null,this.activation=n,this.addBias&&this.variableNames.push(`bias`),this.hasPreluActivationWeights&&this.variableNames.push(`preluActivationWeights`),this.shaderKey=`biasActivation_${n}`}getUserCode(){return`
    ${K(this.activation,this.hasPreluActivationWeights)}
    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${Ma(this.addBias,this.activation)}
        setOutputAtIndex(index, value);
      }
    }
    `}},Xa=class{constructor(e){this.variableNames=[],this.outputShape=[],this.uniforms=`value : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`fill`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        setOutputAtIndex(index, uniforms.value);
      }
    }
  `}};function q(e){let{backend:t,attrs:n}=e,{shape:r,value:i}=n,{dtype:a}=n;if(a||=ft(i),a===`string`){let e=d(a,M(r));return e.fill(i),t.makeTensorInfo(r,a,e)}{let e=new Xa(r),n=[{type:`float32`,data:[i]}];return t.runWebGPUProgram(e,[],a,n)}}var Za={kernelName:dn,backendName:`webgpu`,kernelFunc:q};function J(e){let{inputs:t,attrs:n}=e,{x:r}=t,{shape:i}=n,a=M(r.shape),o=ae(i,a),s=M(o);return k(a===s,()=>`The new shape (${o}) has ${s} elements and the old shape (${r.shape}) has ${a} elements. The new shape and old shape must have the same number of elements.`),e.backend.incRef(r.dataId),{dataId:r.dataId,shape:o,dtype:r.dtype}}var Qa={kernelName:c,backendName:`webgpu`,kernelFunc:J};function $a({a:e,b:t,transposeA:n,transposeB:r,backend:i,bias:a=null,preluActivationWeights:o=null,leakyreluAlpha:s=0,activation:c=null}){let l=e.shape.length,u=t.shape.length,d=n?e.shape[l-2]:e.shape[l-1],f=r?t.shape[u-1]:t.shape[u-2],p=n?e.shape[l-1]:e.shape[l-2],m=r?t.shape[u-2]:t.shape[u-1],h=e.shape.slice(0,-2),g=t.shape.slice(0,-2),_=M(h),v=M(g),y=N(e.shape.slice(0,-2),t.shape.slice(0,-2)).concat([p,m]);k(d===f,()=>`Error in matMul: inner shapes (${d}) and (${f}) of Tensors with shapes ${e.shape} and ${t.shape} and transposeA=${n} and transposeB=${r} must match.`);let b=n?[_,d,p]:[_,p,d],x=r?[v,m,f]:[v,f,m],S=J({inputs:{x:e},backend:i,attrs:{shape:b}}),C=J({inputs:{x:t},backend:i,attrs:{shape:x}}),w=[S,C],T=Math.max(_,v),E=[S,C],ee=[{type:`int32`,data:[p]},{type:`int32`,data:[m]},{type:`int32`,data:[d]}],D,O,te=[T,p,m],ne=j().get(`WEBGPU_MATMUL_PROGRAM_TYPE`);if(ne<0){let e=j().getNumber(`WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL`),t=e>0?e:i.thresholdToIncreaseWorkgroups,n=T*Math.ceil(p/32)*Math.ceil(m/32);ne=n<=t||p<=8&&n<=t*2?T*p*m<=128?H.MatMulReduceProgram:T===1&&f>=2e3?H.MatMulSplitKProgram:H.MatMulSmallOutputSizeProgram:H.MatMulPackedProgram}switch(ne){case H.MatMulReduceProgram:D=new Ga(te,n,r,a,c,o);break;case H.MatMulSplitKProgram:if(O=q({backend:i,attrs:{shape:te,value:0,dtype:e.dtype}}),D=new Ja(te,f,n,r),a||c){O=i.runWebGPUProgram(D,E,e.dtype,ee,O);let t=new Ya(O.shape,a,c,o),n=null,r=[O];a&&r.push(a),o&&r.push(o),c===`leakyrelu`&&(n=[{type:`float32`,data:[s]}],t.uniforms+=` alpha : f32,`);let l=i.runWebGPUProgram(t,r,O.dtype,n);w.push(O);let u=J({inputs:{x:l},backend:i,attrs:{shape:y}});w.push(l);for(let e of w)i.disposeData(e.dataId);return u}break;case H.MatMulSmallOutputSizeProgram:D=new qa(b,x,te,n,r,a,c,o);break;case H.MatMulPackedProgram:D=new Ua(b,te,n,r,a,c,o,i.adapterInfo.isIntel());break;default:throw Error(`Unsupported MatMulProgramType ${ne}.`)}a&&E.push(a),o&&E.push(o),c===`leakyrelu`&&(ee.push({type:`float32`,data:[s]}),D.uniforms+=` alpha : f32,`),O=i.runWebGPUProgram(D,E,e.dtype,ee,O);let re=J({inputs:{x:O},backend:i,attrs:{shape:y}});w.push(O);for(let e of w)i.disposeData(e.dataId);return re}function eo(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a,bias:o,preluActivationWeights:s}=t,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=r;return $a({a:i,b:a,transposeA:c,transposeB:l,backend:n,bias:o,preluActivationWeights:s,leakyreluAlpha:d,activation:u})}var to={kernelName:E,backendName:`webgpu`,kernelFunc:eo},no=class{constructor(e,t,n){this.variableNames=[`AReal`,`AImag`,`BReal`,`BImag`],this.workgroupSize=[128,1,1],this.size=!0,this.outputShape=N(t,n),this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`binaryOpComplex_${e}`,this.op=e}getUserCode(){return`
      fn binaryOpComplex(
          areal : f32, aimag : f32, breal : f32, bimag : f32) -> f32 {
        ${Bi(this.op,!1)}
      }

      ${z(`index`)} {
        if(index < uniforms.size) {
          let areal = getARealByOutputIndex(index);
          let aimag = getAImagByOutputIndex(index);
          let breal = getBRealByOutputIndex(index);
          let bimag = getBImagByOutputIndex(index);
          setOutputAtIndex(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `}},ro=class{constructor(e,t,n){if(this.size=!0,this.variableNames=[`A`,`B`],this.outputShape=N(t,n),this.dispatchLayout=V(this.outputShape),this.op=e,this.useSharedMemoryWithA=t.length<=1&&n.length>1&&t[0]<128,this.useSharedMemoryWithB=n.length<=1&&t.length>1&&n[0]<128,this.useSharedMemoryWithA||this.useSharedMemoryWithB)this.outputComponent=1,this.variableComponents=[1,1],this.lastDimensionSize=this.useSharedMemoryWithB?n[0]:t[0],this.shaderKey=`binary_${e}_${this.lastDimensionSize}`,this.type=`shared`,this.workgroupSize=[256,1,1];else{let r=t.length>0&&t[t.length-1]%4==0,i=n.length>0&&n[n.length-1]%4==0;r&&i?(this.outputComponent=4,this.variableComponents=[4,4]):r&&(St(n)||n[n.length-1]===1)||i&&(St(t)||t[t.length-1]===1)?(this.outputComponent=4,this.variableComponents=r?[4,1]:[1,4]):(this.outputComponent=1,this.variableComponents=[1,1]),this.type=`nonshared`,this.shaderKey=`binary_${e}_${this.variableComponents}`,this.workgroupSize=[128,1,1]}this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.outputComponent,1,1])}getUserCode(){let e,t=this.outputComponent===4?`vec4<f32>`:`f32`,n=`
    fn binaryOperation(a : ${t}, b : ${t}) -> ${t} {
      ${Bi(this.op,this.outputComponent===4)}
    };
    `;if(this.type===`shared`){let t=this.lastDimensionSize>1?`coords[${this.outputShape.length-1}]`:`0`,r=this.useSharedMemoryWithB?`let a = getAByOutputIndex(index);
          let b = sharedBuf[${t}];`:`let a = sharedBuf[${t}];
          let b = getBByOutputIndex(index);`;e=`
        ${n}
        var<workgroup> sharedBuf : array<f32, ${this.lastDimensionSize}>;
        ${z(`index`)} {
          // Fill in the shared memory buffer.
          let localIndex = i32(localId.x);
          if(localIndex < ${this.lastDimensionSize}) {
            sharedBuf[localIndex] = f32(${this.useSharedMemoryWithB?`B`:`A`}[localIndex]);
          }
          workgroupBarrier();

          if(index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            ${r}
            setOutputAtIndex(index, binaryOperation(a, b));
          }
        }
        `}else e=`
       ${n}
       ${z(`index`)} {
         if (index < uniforms.size) {
           let coords = getCoordsFromIndex(index * ${this.outputComponent});
           let a = ${t}(getAByOutputCoords(coords));
           let b = ${t}(getBByOutputCoords(coords));
           setOutputAtIndex(index, binaryOperation(a, b));
         }
       }
       `;return e}};function Y(e){let{inputs:t}=e,{x:n}=t;return e.backend.incRef(n.dataId),{dataId:n.dataId,shape:n.shape,dtype:n.dtype}}var io={kernelName:Ne,backendName:`webgpu`,kernelFunc:Y};function ao(e){let{inputs:t,backend:n}=e,{real:r,imag:i}=t,a=n.makeTensorInfo(r.shape,`complex64`),o=n.tensorMap.get(a.dataId);return o.complexTensorInfos={real:Y({inputs:{x:r},backend:n}),imag:Y({inputs:{x:i},backend:n})},a}var oo={kernelName:fe,backendName:`webgpu`,kernelFunc:ao},so=class{constructor(e,t,n=``){this.variableNames=[`A`],this.size=!0,this.workgroupSize=[128,1,1],this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.op=t,n!==``&&(this.uniforms=n),this.shaderKey=`unary_${t}`}getUserCode(){return`
      fn unaryOperation(a : f32) -> f32 {
        ${G(this.op,!1)}
      }
      ${z(`index`)} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, unaryOperation(a));
        }
      }
      `}};function X({opType:e,cpuKernelImpl:t,dtype:n}){return({inputs:r,backend:i})=>{let{x:a}=r,o=i,s=n||a.dtype;if(o.shouldExecuteOnCPU([a])&&t!=null){let e=t(o.tensorMap.get(a.dataId).values,s);return o.makeTensorInfo(a.shape,s,e)}let c=new so(a.shape,e);return o.runWebGPUProgram(c,[a],s)}}function Z({opType:e,cpuKernelImpl:t,supportsComplex:n=!1,dtype:r}){return({inputs:i,backend:a})=>{let{a:o,b:s}=i,c=a;if(n&&o.dtype===`complex64`){let t=c.tensorMap.get(o.dataId),n=c.tensorMap.get(s.dataId),r,i;if(e!==U.MUL)[r,i]=[[t.complexTensorInfos.real,n.complexTensorInfos.real],[t.complexTensorInfos.imag,n.complexTensorInfos.imag]].map(t=>{let[n,r]=t,i={dataId:n.dataId,dtype:n.dtype,shape:o.shape},a={dataId:r.dataId,dtype:r.dtype,shape:s.shape},l=new ro(e,o.shape,s.shape);return c.runWebGPUProgram(l,[i,a],f(n.dtype,r.dtype))});else{let e=new no(U.COMPLEX_MULTIPLY_REAL,o.shape,s.shape),a=new no(U.COMPLEX_MULTIPLY_IMAG,o.shape,s.shape),l=[{dataId:t.complexTensorInfos.real.dataId,dtype:t.complexTensorInfos.real.dtype,shape:o.shape},{dataId:t.complexTensorInfos.imag.dataId,dtype:t.complexTensorInfos.imag.dtype,shape:o.shape},{dataId:n.complexTensorInfos.real.dataId,dtype:n.complexTensorInfos.real.dtype,shape:s.shape},{dataId:n.complexTensorInfos.imag.dataId,dtype:n.complexTensorInfos.imag.dtype,shape:s.shape}];r=c.runWebGPUProgram(e,l,`float32`),i=c.runWebGPUProgram(a,l,`float32`)}let a=ao({inputs:{real:r,imag:i},backend:c});return c.disposeData(r.dataId),c.disposeData(i.dataId),a}let l=r||f(o.dtype,s.dtype);if((o.dtype===`string`||s.dtype===`string`||c.shouldExecuteOnCPU([o,s]))&&t!=null){let e=c.tensorMap.get(o.dataId).values,n=c.tensorMap.get(s.dataId).values,r=o.dtype===`string`?gr(e):e,i=o.dtype===`string`?gr(n):n,[a,u]=t(o.shape,s.shape,r,i,l);return c.makeTensorInfo(u,l,a)}let u=new ro(e,o.shape,s.shape);return c.runWebGPUProgram(u,[o,s],l)}}var{addImpl:co,castImpl:lo,ceilImpl:uo,concatImpl:fo,equalImpl:po,expImpl:mo,expm1Impl:ho,floorImpl:go,floorDivImpl:_o,gatherNdImpl:vo,gatherV2Impl:yo,greaterEqualImpl:bo,greaterImpl:xo,lessEqualImpl:So,lessImpl:Co,logImpl:wo,maxImpl:To,maximumImpl:Eo,minimumImpl:Do,multiplyImpl:Oo,negImpl:ko,notEqualImpl:Ao,prodImpl:jo,rangeImpl:Mo,rsqrtImpl:No,scatterImpl:Po,simpleAbsImpl:Fo,sliceImpl:Io,stridedSliceImpl:Lo,stringNGramsImpl:Ro,subImpl:zo,tileImpl:Bo,topKImpl:Vo,transposeImpl:Ho,uniqueImpl:Uo}=Qn,Wo={kernelName:`Abs`,backendName:`webgpu`,kernelFunc:X({opType:W.ABS,cpuKernelImpl:Fo})},Go=X({opType:W.ACOS}),Ko={kernelName:ir,backendName:`webgpu`,kernelFunc:Go},qo=X({opType:W.ACOSH}),Jo={kernelName:Kt,backendName:`webgpu`,kernelFunc:qo},Yo={kernelName:`Add`,backendName:`webgpu`,kernelFunc:Z({opType:U.ADD,cpuKernelImpl:co,supportsComplex:!0})},Xo=class{constructor(e){this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e[0],this.variableNames=e.map((e,t)=>`T${t}`),this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.shaderKey=`addN`}getUserCode(){let e=[];this.variableNames.forEach(t=>{e.push(`let v${t} = get${t}ByOutputCoords(coords);`)});let t=this.variableNames.map(e=>`v${e}`).join(` + `);return`
      ${z(`index`)} {
        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            ${e.join(`
        `)}
            setOutputAtIndex(flatIndex, ${t});
          }
        }
      }
    `}};function Zo(e){let{inputs:t,backend:n}=e,r=t;if(r.length===1)return Y({inputs:{x:r[0]},backend:n});let i=r.map(e=>e.dtype).reduce((e,t)=>f(e,t)),a=new Xo(r.map(e=>e.shape));return n.runWebGPUProgram(a,r,i)}var Qo={kernelName:Ln,backendName:`webgpu`,kernelFunc:Zo},$o=class{constructor(e,t){this.variableNames=[`A`],this.workgroupSize=[16,16,1];let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout={x:[0],y:[1]},this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[1,1,1]),this.shaderKey=`transposeShared`}getUserCode(){k(this.workgroupSize[0]===this.workgroupSize[1],()=>`Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`);let e=this.workgroupSize[0];return`
      var<workgroup> tile : array<array<f32, ${this.workgroupSize[0]+1}>, ${this.workgroupSize[0]}>;
      ${z()} {
        var x = i32(workgroupId.x) * ${e} + i32(localId.x);
        var y = i32(workgroupId.y) * ${e} + i32(localId.y);
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] = f32(A[y * width + x]);
        }
        workgroupBarrier();

        x = i32(workgroupId.y) * ${e} + i32(localId.x);
        y = i32(workgroupId.x) * ${e} + i32(localId.y);
        if (x < height && y < width) {
          setOutputAtIndex((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `}},es=class{constructor(e,t){this.variableNames=[`A`],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0;let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.newDim=t,this.shaderKey=`transpose_${t}`}getUserCode(){let e=L(this.outputShape.length),t=ts(this.newDim);return`
      ${z(`index`)} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${this.outputShape.length}D(
              ${e}(${t}), uniforms.aShape)]);
          }
        }
      }
    `}};function ts(e){let t=e.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);let n=Array(t);for(let t=0;t<e.length;t++)n[e[t]]=`coords.${R(t)}`;return n.join()}function Q(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{perm:o}=r,s=n,c=i.shape.length,l=Array(c);for(let e=0;e<l.length;e++)l[e]=i.shape[o[e]];if(n.shouldExecuteOnCPU([i])){let e=s.tensorMap.get(i.dataId).values,t=Ho(e,i.shape,i.dtype,o,l);return n.makeTensorInfo(l,i.dtype,t)}if(i.shape.length===2&&a(o,[1,0])){let e=new $o(i.shape,o);return s.runWebGPUProgram(e,[i],i.dtype)}let u=new es(i.shape,o);return s.runWebGPUProgram(u,[i],i.dtype)}var ns={kernelName:zt,backendName:`webgpu`,kernelFunc:Q},rs=class{constructor(e,t,r){this.variableNames=[`x`],this.uniforms=`reduceSize : i32,`,this.size=!0,this.inputShape=[e.batchSize,e.inSize];let[i]=n(this.inputShape,[1]);this.outputShape=i.length===0?[1]:i,this.workgroupSize=e.inSize>=32768&&r>=512?[512,1,1]:e.inSize>=4096?[256,1,1]:[64,1,1],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,[1,1,1]),this.reduceType=t,this.shaderKey=`reduce_${t}`}getUserCode(){let e=``,t=`0.0`,n=this.workgroupSize[0];this.reduceType===`min`||this.reduceType===`max`?(e=`
         if (isnan(candidate)) {
          bestValue = uniforms.NAN;
         } else if (!isnan(bestValue) && candidate ${this.reduceType===`min`?`<`:`>`} bestValue)
           {  bestValue = candidate; }`,t=`f32(x[offset])`):this.reduceType===`sum`||this.reduceType===`mean`?e=` bestValue = bestValue + candidate; `:this.reduceType===`prod`?(e=` bestValue = bestValue * candidate; `,t=`1.0`):this.reduceType===`all`?(e=` bestValue = f32(bestValue >= 1.0 && candidate >= 1.0); `,t=`1.0`):this.reduceType===`any`&&(e=` bestValue = f32(bestValue >= 1.0 || candidate >= 1.0); `,t=`0.0`);let r=this.reduceType===`mean`?`setOutputAtIndex(outputIndex, bestValue / f32(uniforms.reduceSize));`:`setOutputAtIndex(outputIndex, bestValue);`;return`
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }

       ${`
         var<workgroup> xBestValues : array<f32, ${n}>;
       `}
       fn getOffset(outputIndex : i32) -> i32 {
         let outputCoords = getCoordsFromIndex(outputIndex);
         let offset = ${this.outputShape.length===1?`outputCoords`:`outputCoords[0]`} * uniforms.reduceSize;
          return offset;
       }
       ${z(`index`)} {
         let outputIndex = index / ${n};
         let offset = getOffset(outputIndex);
         var bestValue = ${t};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(u32(Length), ${n}u);
         for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
             k = k + ${n}) {
           let candidate = f32(x[offset + k]);
           ${e}
         }
         xBestValues[localId.x] = bestValue;
         workgroupBarrier();

         var reduceSize = min(u32(Length), ${n}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            ${e}
            xBestValues[localId.x] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (localId.x == 0u && outputIndex < uniforms.size) {
          ${r}
        }
       }
     `}},is={mean:`float32`,all:`bool`,any:`bool`};function $(e,t,r,i,a){let o=e.shape.length,s=[],c=An(t,e.shape),l=c,u=nr(l,o),d=e;u!=null&&(d=Q({inputs:{x:e},attrs:{perm:u},backend:a}),l=Nn(l.length,o),s.push(d)),Be(i,l,o);let[f,p]=n(d.shape,l),m=f;r&&(m=$t(f,c));let h;if((i===`max`||i===`prod`)&&a.shouldExecuteOnCPU([d])){let t=a.tensorMap.get(d.dataId).values;switch(i){case`max`:let n=To(t,M(p),m,e.dtype);h=a.makeTensorInfo(m,e.dtype,n);break;case`prod`:let{outVals:r,outShape:o,outDtype:s}=jo(d.shape,d.dtype,t,l);h=a.makeTensorInfo(o,s,r);break;default:throw Error(`${i} CPU implementation is not yet supported.`)}}else{let t=M(p),n={windowSize:t,inSize:t,batchSize:M(d.shape)/t,outSize:1},r=is[i]||wr(e.dtype),o=[{type:`int32`,data:[t]}],c=new rs(n,i,a.device.limits.maxComputeWorkgroupSizeX),l=a.runWebGPUProgram(c,[d],r,o);s.push(l),h=J({inputs:{x:l},attrs:{shape:m},backend:a})}return s.forEach(e=>a.disposeData(e.dataId)),h}function as(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{keepDims:a,axis:o}=r;return $(i,o,a,`all`,n)}var os={kernelName:`All`,backendName:`webgpu`,kernelFunc:as};function ss(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{keepDims:a,axis:o}=r;return $(i,o,a,`any`,n)}var cs={kernelName:`Any`,backendName:`webgpu`,kernelFunc:ss},ls=class{constructor(e,t,r){this.workgroupSize=[64,1,1],this.variableNames=[`x`],this.uniforms=`infinityValue : f32,`,this.size=!0;let i=[t];this.op=r===`min`?`<`:`>`;let[a,o]=n(e,i);this.outputShape=a.length===0?[1]:a,this.dispatchLayout=V(this.outputShape),M(o)<32?(this.type=`plain`,this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize)):(this.type=`shared`,this.dispatch=B(this.dispatchLayout,this.outputShape,[1,1,1])),this.inputShape=e,this.shaderKey=`argMinMax_${this.op}_${this.type}`}getUserCode(){let e=this.workgroupSize[0],t=()=>this.inputShape.length===1?`uniforms.xShape`:`uniforms.xShape.${R(this.inputShape.length-1)}`,n=()=>{let e=``;if(this.outputShape.length===1)this.inputShape.length!==1&&(e+=`outputCoords,`);else for(let t=0;t<this.outputShape.length;t++)e+=`outputCoords.${R(t)},`;return e};return this.type===`shared`?`
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      ${`
      var<workgroup> xBestIndices : array<i32, ${e}>;
      var<workgroup> xBestValues : array<f32, ${e}>;
    `}

      ${z(`index`)} {
        let outputIndex = index / ${e};
        let reduceLength = ${t()};

        var bestIndex = i32(localId.x);
        var bestValue = uniforms.infinityValue;
        let outputCoords = getCoordsFromIndex(outputIndex);
        for (var k = i32(localId.x); k < reduceLength && outputIndex < uniforms.size;
            k = k + ${e}) {
          let candidate = getX(${n()} k);
          if (!isnan(candidate) && candidate ${this.op} bestValue) {
            bestValue = candidate;
            bestIndex = k;
          }
        }
        xBestValues[localId.x] = bestValue;
        xBestIndices[localId.x] = bestIndex;
        workgroupBarrier();

        var reduceSize = min(u32(reduceLength), ${e}u);
        for (var currentSize = reduceSize / 2u; reduceSize > 1u;
            currentSize = reduceSize / 2u) {
          let interval = DIV_CEIL(reduceSize, 2u);
          if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              xBestValues[localId.x] = bestValue;
              xBestIndices[localId.x] = xBestIndices[localId.x + interval];
            }
          }
          reduceSize = interval;
          workgroupBarrier();
        }

        if (localId.x == 0u && outputIndex < uniforms.size) {
          setOutputAtIndexI32(outputIndex, xBestIndices[localId.x]);
        }
      }
    `:`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let outputCoords = getCoordsFromIndex(index);
          var bestIndex = 0;
          var bestValue = getX(${n()} 0);
          let reduceLength = ${t()};
          for (var i = 1; i < reduceLength; i++) {
            let candidate = getX(${n()} i);
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
          setOutputAtIndexI32(index, bestIndex);
        }
      }
      `}};function us(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=An(a,i.shape),s=nr(o,i.shape.length),c=i,l=[];s!=null&&(c=Q({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=Nn(o.length,c.shape.length)),Be(`argMax`,[o[0]],c.shape.length);let u=new ls(c.shape,o[0],`max`),d=n.runWebGPUProgram(u,[c],`int32`,[{type:`float32`,data:[-1/0]}]);return l.forEach(e=>n.disposeData(e.dataId)),d}var ds={kernelName:an,backendName:`webgpu`,kernelFunc:us};function fs(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a}=r,o=An(a,i.shape),s=nr(o,i.shape.length),c=i,l=[];s!=null&&(c=Q({inputs:{x:i},backend:n,attrs:{perm:s}}),l.push(c),o=Nn(o.length,c.shape.length)),Be(`argMin`,[o[0]],c.shape.length);let u=new ls(c.shape,o[0],`min`),d=n.runWebGPUProgram(u,[c],`int32`,[{type:`float32`,data:[1/0]}]);return l.forEach(e=>n.disposeData(e.dataId)),d}var ps={kernelName:At,backendName:`webgpu`,kernelFunc:fs},ms=X({opType:W.ASIN}),hs={kernelName:or,backendName:`webgpu`,kernelFunc:ms},gs=X({opType:W.ASINH}),_s={kernelName:_r,backendName:`webgpu`,kernelFunc:gs},vs=X({opType:W.ATAN}),ys={kernelName:Rt,backendName:`webgpu`,kernelFunc:vs},bs=Z({opType:U.ATAN2}),xs={kernelName:pr,backendName:`webgpu`,kernelFunc:bs},Ss=X({opType:W.ATANH}),Cs={kernelName:qe,backendName:`webgpu`,kernelFunc:Ss},ws=class{constructor(e){this.variableNames=[`x`],this.uniforms=`strides : vec2<i32>,`,this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`poolWithFilterSizeEqualsOne`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];

          let xRCCorner = coords.yz * uniforms.strides;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutputAtIndex(index, value);
        }
      }
    `}},Ts=class{constructor(e,t,n=!1,r=!1,i=!1){if(this.variableNames=[`x`],this.uniforms=`strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,`,this.workgroupSize=[128,1,1],this.size=!0,t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.poolType=t,this.computePositions=n,this.flattenPositions=r,this.includeBatchIndex=i,this.shaderKey=`pool2D_${t}_${n}_${r}_${i}`}getUserCode(){let e;e=this.poolType===`avg`?`resultValue = resultValue + value; count = count + 1.0;`:this.computePositions?`let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${this.flattenPositions?this.includeBatchIndex?`((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d`:`(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d`:`wR * uniforms.filterDims.y + wC`};
      }`:`resultValue = max(value, resultValue);`;let t=`resultValue`;return this.poolType===`avg`&&(t=`resultValue / max(count, 1.0)`),`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.strides - uniforms.pads;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          ${this.computePositions?`var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;`:`var resultValue = ${this.poolType===`avg`?`0.0`:`-1.0 / pow(10.0, -20.0)`};`}

          var count = 0.0;
          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilations.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilations.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, d);
              ${e}
            }
          }

          ${this.computePositions?`setOutputAtIndexI32(index, maxPosition);`:`setOutputAtIndex(index, ${t});`}
        }
      }
    `}},Es=class{constructor(e,t,n=!1,r=!1,i=!1){if(this.variableNames=[`x`],this.uniforms=`strides : vec3<i32>, pads : vec3<i32>, convDims : vec3<i32>, filterDims : vec3<i32>,`,this.workgroupSize=[128,1,1],this.size=!0,t===`avg`&&n)throw Error(`Cannot compute positions for average pool.`);this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.poolType=t,this.computePositions=n,this.flattenPositions=r,this.includeBatchIndex=i,this.shaderKey=`pool3D_${t}_${n}_${r}_${i}`}getUserCode(){let e;e=this.poolType===`avg`?`resultValue += value; count += 1.0;`:this.computePositions?`let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${this.flattenPositions?this.includeBatchIndex?`(((batch * uniforms.xShape.y + xD) * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch`:`((xD * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch`:`wD * uniforms.filterDims.y * uniforms.filterDims.y + wR * uniforms.filterDims.z + wC`};
      }`:`resultValue = max(value, resultValue);`;let t=`resultValue`;return this.poolType===`avg`&&(t=`resultValue / max(count, 1.0)`),`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords.x;
          let ch = coords.u;

          let xCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
          let xDCorner = xCorner.x;
          let xRCorner = xCorner.y;
          let xCCorner = xCorner.z;

          ${this.computePositions?`var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;`:`var resultValue = ${this.poolType===`avg`?`0.0`:`-1.0 / pow(10.0, -20.0)`};`}

          var count = 0.0;
          for (var wD = 0; wD < uniforms.filterDims.x; wD++) {
            let xD = xDCorner + wD;
            if (xD < 0 || xD >= uniforms.convDims.x) {
              continue;
            }

            for (var wR = 0; wR < uniforms.filterDims.y; wR++) {
              let xR = xRCorner + wR;
              if (xR < 0 || xR >= uniforms.convDims.y) {
                continue;
              }

              for (var wC = 0; wC < uniforms.filterDims.z; wC++) {
                let xC = xCCorner + wC;
                if (xC < 0 || xC >= uniforms.convDims.z) {
                  continue;
                }

                let value = getX(batch, xD, xR, xC, ch);
                ${e}
              }
            }
          }

          ${this.computePositions?`setOutputAtIndexI32(index, maxPosition);`:`setOutputAtIndex(index, ${t});`}
        }
      }
    `}};function Ds(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reductionIndices:a,keepDims:o}=r;return $(i,a,o,`max`,n)}var Os={kernelName:`Max`,backendName:`webgpu`,kernelFunc:Ds};function ks(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{keepDims:a,axis:o}=r;return $(i,o,a,`mean`,n)}var As={kernelName:rr,backendName:`webgpu`,kernelFunc:ks};function js(e,t,n,r){if(t.filterWidth===1&&t.filterHeight===1&&a(t.inShape,t.outShape))return Y({inputs:{x:e},backend:r});if(t.filterWidth===t.inWidth&&t.filterHeight===t.inHeight&&t.batchSize===1&&t.padInfo.type===`VALID`){let i=e.shape.length,a=J({inputs:{x:e},backend:r,attrs:{shape:[e.shape[i-3]*e.shape[i-2],e.shape[i-1]]}}),o;n===`avg`?o=ks({inputs:{x:a},backend:r,attrs:{axis:0,keepDims:!1}}):(k(n===`max`,()=>`Invalid pool type ${n}`),o=Ds({inputs:{x:a},backend:r,attrs:{reductionIndices:0,keepDims:!1}}));let s=J({inputs:{x:o},backend:r,attrs:{shape:t.outShape}});return r.disposeData(a.dataId),r.disposeData(o.dataId),s}let i,o=[{type:`int32`,data:[t.strideHeight,t.strideWidth]}];return t.filterHeight===1&&t.filterWidth===1?i=new ws(t):(n===`avg`?i=new Ts(t,`avg`):(k(n===`max`,()=>`Invalid pool type ${n}`),i=new Ts(t,`max`)),o.push({type:`int32`,data:[t.padInfo.top,t.padInfo.left]},{type:`int32`,data:[t.dilationHeight,t.dilationWidth]},{type:`int32`,data:[t.inHeight,t.inWidth]},{type:`int32`,data:[t.effectiveFilterHeight,t.effectiveFilterWidth]})),r.runWebGPUProgram(i,[e],e.dtype,o)}function Ms(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;return js(i,lr(i.shape,a,o,1,s,c),`avg`,n)}var Ns={kernelName:g,backendName:`webgpu`,kernelFunc:Ms};function Ps(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dataFormat:c,dimRoundingMode:l}=r,u=br(i.shape,a,o,[1,1,1],s,l,c),d=new Es(u,`avg`),f=[{type:`int32`,data:[u.strideDepth,u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.padInfo.front,u.padInfo.top,u.padInfo.left]},{type:`int32`,data:[u.inDepth,u.inHeight,u.inWidth]},{type:`int32`,data:[u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth]}];return n.runWebGPUProgram(d,[i],i.dtype,f)}var Fs={kernelName:ur,backendName:`webgpu`,kernelFunc:Ps},Is=class{constructor(e){this.variableNames=[`dy`],this.uniforms=`strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, avgMultiplier : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`avgPool2DBackprop`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d = coords[3];

        let dyRCCorner = vec2<i32>(coords.yz) - uniforms.pads;
        let dyRCorner = dyRCCorner.x;
        let dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR = wR + uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC = wC + uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);

            dotProd = dotProd + dyValue * uniforms.avgMultiplier;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `}},Ls=class{constructor(e){this.variableNames=[`dy`],this.uniforms=`strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
       outDepth : i32, outHeight : i32, outWidth : i32, avgMultiplier : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`avgPool3DBackprop`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let ch = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyDCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wD = 0; wD < uniforms.filterDims[0]; wD++) {
          let dyD = f32(dyDCorner + wD) / f32(uniforms.strides[0]);

          if (dyD < 0.0 || dyD >= f32(uniforms.outDepth) || fract(dyD) > 0.0) {
            continue;
          }
          let idyD = i32(dyD);

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let dyValue = getDy(batch, idyD, idyR, idyC, ch);
              dotProd += dyValue * uniforms.avgMultiplier;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `}};function Rs(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=br(o.shape,s,c,1,l,u),f=new Ls(d),p=1/(d.filterDepth*d.filterHeight*d.filterWidth),m=[{type:`int32`,data:[d.strideDepth,d.strideHeight,d.strideWidth]},{type:`int32`,data:[d.effectiveFilterDepth-1-d.padInfo.front,d.effectiveFilterHeight-1-d.padInfo.top,d.effectiveFilterWidth-1-d.padInfo.left]},{type:`int32`,data:[d.effectiveFilterDepth,d.effectiveFilterHeight,d.effectiveFilterWidth]},{type:`int32`,data:[d.outDepth]},{type:`int32`,data:[d.outHeight]},{type:`int32`,data:[d.outWidth]},{type:`float32`,data:[p]}];return n.runWebGPUProgram(f,[i],o.dtype,m)}var zs={kernelName:Ze,backendName:`webgpu`,kernelFunc:Rs};function Bs(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a;si([i,a],`avgPoolGrad`);let{filterSize:s,strides:c,pad:l}=r,u=lr(o.shape,s,c,1,l),d=new Is(u),f=1/(u.filterHeight*u.filterWidth),p=[{type:`int32`,data:[u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.effectiveFilterHeight-1-u.padInfo.top,u.effectiveFilterWidth-1-u.padInfo.left]},{type:`int32`,data:[u.dilationHeight,u.dilationWidth]},{type:`int32`,data:[u.effectiveFilterHeight,u.effectiveFilterWidth]},{type:`int32`,data:[u.outHeight]},{type:`int32`,data:[u.outWidth]},{type:`float32`,data:[f]}];return n.runWebGPUProgram(d,[i],o.dtype,p)}var Vs={kernelName:T,backendName:`webgpu`,kernelFunc:Bs};function Hs(e){let{inputs:t,backend:n,attrs:r}=e,{a:i,b:a}=t,{transposeA:o,transposeB:s}=r;return $a({a:i,b:a,transposeA:o,transposeB:s,backend:n})}var Us={kernelName:x,backendName:`webgpu`,kernelFunc:Hs},Ws=class{constructor(e,t){this.variableNames=[`source`],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.rank=t.length,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.start=e,this.uniforms=`start : ${L(e.length)}, `,this.shaderKey=`slice`}getUserCode(){let e=L(this.rank),t=Ks(this.rank),n;return n=this.start.length===1?this.outputShape.map((e,t)=>`sourceLoc = uniforms.start + coords;`):this.outputShape.map((e,t)=>`sourceLoc.${Gs[t]} = uniforms.start.${R(t)} + coords.${Gs[t]};`),`
      ${z(`index`)} {
        if (index < uniforms.size) {
          var sourceLoc : ${e};
          let coords = getCoordsFromIndex(index);
          ${n.join(`
`)}
          setOutputAtIndex(index, getSource(${t}));
        }
      }
    `}},Gs=[`x`,`y`,`z`,`w`,`u`,`v`];function Ks(e){if(e===1)return`sourceLoc`;if(e<=6)return Gs.slice(0,e).map(e=>`sourceLoc.${e}`).join(`,`);throw Error(`Slicing for rank ${e} is not yet supported`)}function qs(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,size:o}=r,[s,c]=Pn(i,a,o);if(jn(i,s,c),n.shouldExecuteOnCPU([i])||i.dtype===`string`){let e=Io(n.tensorMap.get(i.dataId).values,s,c,i.shape,i.dtype);return n.makeTensorInfo(c,i.dtype,e)}if(M(c)===0)return n.makeTensorInfo(c,i.dtype,[]);let l=new Ws(s,c),u=[{type:`int32`,data:s}];return n.runWebGPUProgram(l,[i],i.dtype,u)}var Js={kernelName:Un,backendName:`webgpu`,kernelFunc:qs},Ys={kernelName:Me,backendName:`webgpu`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,crops:o}=r;k(i.shape.length<=4,()=>`batchToSpaceND for rank > 4 with a WebGPU backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=_e(i.shape,a,s),l=Wn(c.length,a.length),u=Et(i.shape,a,s),d=bt(o,a.length),f=kt(u,o,a.length),p=[],m=J({inputs:{x:i},backend:n,attrs:{shape:c}}),h=Q({inputs:{x:m},backend:n,attrs:{perm:l}}),g=J({inputs:{x:h},backend:n,attrs:{shape:u}}),_=qs({inputs:{x:g},backend:n,attrs:{begin:d,size:f}});return p.push(m),p.push(h),p.push(g),p.forEach(e=>n.disposeData(e.dataId)),_}},Xs=`
  fn bincount_write(index: i32, value: f32) {
    ${F(`&result[index]`,`value`,`float32`)}
  }
`,Zs=`
  fn bincount_write(index: i32, value: f32) {
    atomicStore(&result[index], bitcast<i32>(value));
  }
`,Qs=class{constructor(e,t,n=!1){this.outputShape=[],this.variableNames=[`x`],this.uniforms=`binCountSize : i32,`,this.workgroupSize=[64,1,1],this.atomic=!0,this.hasWeights=!0,this.binaryOutput=!1,this.outputShape=e,this.rank=e.length,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.binaryOutput=n,n&&(this.atomic=!1),this.hasWeights=t,this.hasWeights&&this.variableNames.push(`w`),this.shaderKey=`bincount_${this.hasWeights}_${this.binaryOutput}_${this.rank}`}getUserCode(){return`
    ${this.binaryOutput?Zs:Xs}
  ${z(`index`)} {
    ${this.rank===1?`if (index < uniforms.xShape) {
      let indexVal = i32(getX(index));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput?1:this.hasWeights?`getW(index)`:`1.`};
        bincount_write(indexVal, value);
      }
    }`:`let coord = getCoordsFromIndex(index);
    if (coordsInBounds2D(coord, uniforms.xShape)) {
      let indexVal = i32(getX(coord[0], coord[1]));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput?1:this.hasWeights?`getW(coord[0], coord[1])`:`1.`};
        bincount_write(coord.x * uniforms.binCountSize + indexVal, value);
      }
    }`}
  }
  `}};function $s(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o}=r,s=M(i.shape),c=M(a.shape)>0,l=[o],u=a.dtype,d=q({backend:n,attrs:{shape:l,value:0,dtype:u}}),f=new Qs([s],c),p=[{type:`int32`,data:[o]}],m=c?[i,a]:[i];return n.runWebGPUProgram(f,m,u,p,d)}var ec={kernelName:bn,backendName:`webgpu`,kernelFunc:$s},tc=class{constructor(e){this.outputShape=[],this.variableNames=[`s0`,`s1`],this.uniforms=`s0Size : i32, s1Size : i32, `,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`broadcastArgs`}getUserCode(){return`
  ${z(`index`)} {
    if (index < uniforms.size) {
      var s0 = 1.0;
      var s1 = 1.0;
      let indexS0 = index - uniforms.size + uniforms.s0Size;
      let indexS1 = index - uniforms.size + uniforms.s1Size;
      if (indexS0 >= 0) {
        s0 = getS0(indexS0);
      }
      if (indexS1 >= 0) {
        s1 = getS1(indexS1);
      }

      if (s0 == 1.0) {
        setOutputAtIndex(index, s1);
      } else if (s1 == 1.0) {
        setOutputAtIndex(index, s0);
      } else if (s0 != s1) {
        setOutputAtIndex(index, uniforms.NAN);
      } else {
        setOutputAtIndex(index, s0);
      }
    }
  }
  `}};function nc(e){let{inputs:t,backend:n}=e,{s0:r,s1:i}=t;if(n.shouldExecuteOnCPU([r,i])){let e=n.tensorMap.get(r.dataId),t=n.tensorMap.get(i.dataId),a=e.values,o=t.values,s=N(Array.from(a),Array.from(o));return n.makeTensorInfo([s.length],`int32`,Int32Array.from(s))}let a=M(r.shape),o=M(i.shape),s=new tc(Math.max(a,o)),c=[{type:`int32`,data:[a]},{type:`int32`,data:[o]}];return n.runWebGPUProgram(s,[r,i],`int32`,c)}var rc={kernelName:_n,backendName:`webgpu`,kernelFunc:nc},ic=Z({opType:U.NOT_EQUAL,dtype:`bool`,cpuKernelImpl:Ao}),ac={kernelName:rn,backendName:`webgpu`,kernelFunc:ic};function oc(e){let{inputs:t,backend:n}=e,{input:r}=t;return Y({inputs:{x:n.tensorMap.get(r.dataId).complexTensorInfos.real},backend:n})}var sc={kernelName:tt,backendName:`webgpu`,kernelFunc:oc};function cc(e,t){let n=new so(e.shape,W.TO_INT),r=t.runWebGPUProgram(n,[e],`int32`);return{dataId:r.dataId,shape:r.shape,dtype:r.dtype}}function lc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dtype:a}=r;if(a===`complex64`){if(i.dtype===`complex64`)return Y({inputs:{x:i},backend:n});let e=Ce(i.shape),t=lc({inputs:{x:i},backend:n,attrs:{dtype:`float32`}}),r=ao({inputs:{real:t,imag:e},backend:n});return e.dispose(),n.disposeData(t.dataId),r}if(i.dtype===`complex64`){let e=oc({inputs:{input:i},backend:n}),t=lc({inputs:{x:e},backend:n,attrs:{dtype:a}});return n.disposeData(e.dataId),t}if(!ce(i.dtype,a)){let e=Y({inputs:{x:i},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:a}}if(n.shouldExecuteOnCPU([i])){let e=n.tensorMap.get(i.dataId).values,[t,r,o]=lo(e,i.shape,i.dtype,a);return n.makeTensorInfo(t,r,o)}if(a===`int32`)return cc(i,n);if(a===`bool`){let e=n.makeTensorInfo([],`bool`,ct(`bool`,1)),t=ic({inputs:{a:i,b:e},backend:n});return n.disposeData(e.dataId),t}throw Error(`Error in Cast: failed to cast ${i.dtype} to ${a}`)}var uc={kernelName:Ae,backendName:`webgpu`,kernelFunc:lc},dc=X({opType:W.CEIL,cpuKernelImpl:uo}),fc={kernelName:Ie,backendName:`webgpu`,kernelFunc:dc},pc=class{constructor(e){this.variableNames=[`A`],this.uniforms=`minVal : f32, maxVal : f32,`,this.workPerThread=4,this.workgroupSize=[64,1,1],this.outputComponent=4,this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.shaderKey=`clipVec4`}getUserCode(){return`
      ${z(`index`)} {
        if(index < uniforms.size) {
          let value = getAByOutputIndex(index);
          var clampedValue = clamp(
              value, vec4<f32>(uniforms.minVal), vec4<f32>(uniforms.maxVal));
          clampedValue = select(clampedValue, value, isnanVec4(value));
          setOutputAtIndex(index, clampedValue);
        }
      }
    `}},mc=class{constructor(e){this.variableNames=[`A`],this.uniforms=`minVal : f32, maxVal : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`clip`}getUserCode(){return`
      ${z(`index`)} {
        if(index < uniforms.size) {
          let value = getAByOutputIndex(index);
          if (isnan(value)) {
            setOutputAtIndex(index, value);
            return;
          }
          setOutputAtIndex(index, clamp(value, uniforms.minVal, uniforms.maxVal));
        }
      }
    `}};function hc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{clipValueMin:a,clipValueMax:o}=r,s,c=[{type:`float32`,data:[a]},{type:`float32`,data:[o]}];return s=M(i.shape)%4==0?new pc(i.shape):new mc(i.shape),n.runWebGPUProgram(s,[i],i.dtype,c)}var gc={kernelName:te,backendName:`webgpu`,kernelFunc:hc},_c=class{constructor(e){this.outputShape=[],this.variableNames=[`real`,`imag`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`complexAbs`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let re = abs(getRealByOutputIndex(index));
        let im = abs(getImagByOutputIndex(index));
        let mx = max(re, im);

        // The length function in wgsl may be not underflow-safe on some GPUs.
        // So the safe solution is to ensure underflow-safety in all cases.
        setOutputAtIndex(index, select(mx * length(vec2<f32>(1, min(re, im)/mx)), 0.0, mx == 0.0));
      }
    }
  `}};function vc(e,t){return{dataId:t.dataId,dtype:t.dtype,shape:e.shape}}function yc(e){let{inputs:t,backend:n}=e,{x:r}=t,i=n.tensorMap.get(r.dataId),a=new _c(r.shape),o=[vc(r,i.complexTensorInfos.real),vc(r,i.complexTensorInfos.imag)];return n.runWebGPUProgram(a,o,o[0].dtype)}var bc={kernelName:Se,backendName:`webgpu`,kernelFunc:yc},xc=class{constructor(e){this.uniforms=``,this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=er(e,1),this.variableNames=e.map((e,t)=>`T${t}`),this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.offsetLength=e.length-1;for(let e=0;e<this.offsetLength;e++)this.uniforms+=`offset${e} : i32,`;this.shaderKey=`concat`}getUserCode(){let e=[];if(this.offsetLength>0){e.push(`if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }`);for(let t=1;t<this.offsetLength;t++)e.push(`else if (yC < uniforms.offset${[t]}){ setOutputAtCoords(coords.x, coords.y, getT${t}(yR, yC - uniforms.offset${t-1})); }`);let t=this.offsetLength,n=this.offsetLength-1;e.push(`else { setOutputAtCoords(coords.x, coords.y, getT${t}(yR, yC - uniforms.offset${n})); }`)}else e.push(`setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));`);return`
      ${z(`index`)} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${e.join(`
        `)}
          }
        }
      }
    `}};function Sc(e){let{inputs:t,backend:n}=e,{input:r}=t;return Y({inputs:{x:n.tensorMap.get(r.dataId).complexTensorInfos.imag},backend:n})}var Cc={kernelName:xn,backendName:`webgpu`,kernelFunc:Sc};function wc(e,t,n){let r=e[0].dtype;if(r===`complex64`){let r=e.map(e=>oc({inputs:{input:e},backend:n})),i=e.map(e=>Sc({inputs:{input:e},backend:n})),a=wc(r,t,n),o=wc(i,t,n),s=ao({inputs:{real:a,imag:o},backend:n});return r.forEach(e=>n.disposeData(e.dataId)),i.forEach(e=>n.disposeData(e.dataId)),n.disposeData(a.dataId),n.disposeData(o.dataId),s}let i=n.shouldExecuteOnCPU(e);if(r===`string`&&(i=!0),i){let i=e.map(e=>{let r=[-1,M(e.shape.slice(t))];return J({inputs:{x:e},backend:n,attrs:{shape:r}})}),a=fo(i.map(e=>({vals:n.readSync(e.dataId),shape:e.shape})),er(i.map(e=>e.shape),1),r,i[0].shape[0]===1),o=er(e.map(e=>e.shape),t),s=n.makeTensorInfo(o,r,a);return i.forEach(e=>n.disposeData(e.dataId)),s}let a=n.device.limits.maxStorageBuffersPerShaderStage-1;if(e.length>a){let r=[];for(let i=0;i<e.length;i+=a){let o=e.slice(i,i+a);r.push(wc(o,t,n))}let i=wc(r,t,n);for(let e of r)n.disposeData(e.dataId);return i}let{tensors2D:o,outShape:s}=Tc(e,t,n),c=o.map(e=>e.shape),l=new xc(c),u=[],d=Array(c.length-1);if(d.length>0){d[0]=c[0][1],u.push({type:`int32`,data:[d[0]]});for(let e=1;e<d.length;e++)d[e]=d[e-1]+c[e][1],u.push({type:`int32`,data:[d[e]]})}let f=n.runWebGPUProgram(l,o,o[0].dtype,u);o.forEach(e=>n.disposeData(e.dataId));let p=J({inputs:{x:f},backend:n,attrs:{shape:s}});return n.disposeData(f.dataId),p}function Tc(e,t,n){let r=er(e.map(e=>e.shape),t);return{tensors2D:e.map(e=>J({inputs:{x:e},backend:n,attrs:{shape:[M(e.shape.slice(0,t)),M(e.shape.slice(t))]}})),outShape:r}}function Ec(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r,a=An(i,t[0].shape)[0],o=t.map(e=>e.shape);Zt(o,a);let s=er(t.map(e=>e.shape),a);if(M(s)===0)return n.makeTensorInfo(s,t[0].dtype,[]);let c=t.filter(e=>M(e.shape)>0);return c.length===1?Y({inputs:{x:c[0]},backend:n}):wc(c,a,n)}var Dc={kernelName:Ue,backendName:`webgpu`,kernelFunc:Ec};function Oc(e,t,n,r,i=!1,a=null,o=!1,s=4,c=4,l=4){let u=e=>{switch(e){case 1:return`resData = f32(x[xIndex]);`;case 3:return`resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);`;case 4:return`resData = vec4<f32>(x[xIndex / 4]);`;default:throw Error(`innerElementSize ${e} is not supported.`)}},d=e=>{switch(e){case 1:return`return f32(W[row * uniforms.wShape[3] + col]);`;case 4:return`return vec4<f32>(W[(row * uniforms.wShape[3] + col) / 4]);`;default:throw Error(`innerElementSize ${e} is not supported.`)}},f=e?`
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      `:`
      let coord = vec4<i32>(batch, xCh, xRow, xCol);
      `,p=e?`
      let coords = vec4<i32>(
        batch,
        row / outWidth,
        row % outWidth,
        col);
      `:`
      let coords = vec4<i32>(
        batch,
        row,
        col / outWidth,
        col % outWidth);
      `,m=e?`uniforms.xShape[1]`:`uniforms.xShape[2]`,h=e?`uniforms.xShape[2]`:`uniforms.xShape[3]`,g=e?`row`:`col`,_=e?`col`:`row`,v=`
      let inChannels = uniforms.wShape[2];
      let outWidth = ${e?`uniforms.outShape[2]`:`uniforms.outShape[3]`};
      let outRow = ${g} / outWidth;
      let outCol = ${g} % outWidth;

      let WRow = ${_} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${_} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
      let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
      let xCh = ${_} % inChannels;
      var resData = ${I(s)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${m} && xCol >= 0 && xCol < ${h}) {
        ${f}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${u(s)}
      }
      return resData;`,y=e?t&&r?`
      ${v}`:`
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${v}
      }
      return ${I(s)}(0.0);`:r&&n?`
      ${v}`:`
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${v}
      }
      return ${I(s)}(0.0);`,b=`${d(c)}`,x=I(l),S=I(e?s:c),C=I(e?c:s);return`
      ${K(a,o,l===4,4)}
      fn mm_readA(batch: i32, row : i32, col : i32) -> ${S} {
        ${e?y:b}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> ${C} {
        ${e?b:y}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : ${x}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        var value = valueIn;
        let outWidth = ${e?`uniforms.outShape[2]`:`uniforms.outShape[3]`};
        ${p}
        ${Ma(i,a)}
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }`}var kc=class{constructor(e,t,n,r,i=!1,a=null,o=!1,s=!1){this.variableNames=[`x`,`W`],this.uniforms=`filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat===`channelsLast`,this.isVec4=((e.inChannels%4==0||e.inChannels%3==0)&&this.isChannelsLast||e.outWidth%4==0&&!this.isChannelsLast)&&e.outChannels%4==0,this.dispatchLayout=this.isChannelsLast?{x:[3],y:[1,2],z:[0]}:{x:[2,3],y:[1],z:[0]},this.workgroupSize=ri(this.dispatchLayout,this.outputShape,this.isVec4),this.elementsPerThread=ii(this.dispatchLayout,this.outputShape,this.isVec4),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread),this.isVec4?(this.outputComponent=4,this.isChannelsLast&&e.inChannels%4!=0?(this.innerElementSize=3,this.variableComponents=[1,4]):(this.innerElementSize=4,this.variableComponents=[4,4]),i&&(this.variableNames.push(`bias`),this.variableComponents.push(4)),o&&(this.variableNames.push(`preluActivationWeights`),this.variableComponents.push(4))):(this.innerElementSize=this.elementsPerThread[0],i&&this.variableNames.push(`bias`),o&&this.variableNames.push(`preluActivationWeights`)),this.sequentialAccessByThreads=s,this.addBias=i,this.activation=a,this.hasPreluActivationWeights=o,this.tileAOuter=this.workgroupSize[1]*this.elementsPerThread[1],this.tileBOuter=this.workgroupSize[0]*this.elementsPerThread[0],this.tileInner=Math.max(this.workgroupSize[0]*this.innerElementSize,this.workgroupSize[1]),this.fitAOuter=t%this.tileAOuter===0,this.fitBOuter=n%this.tileBOuter===0,this.fitInner=r%this.tileInner===0,this.shaderKey=`conv2DMM_${this.elementsPerThread}_${this.activation}}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.innerElementSize}_${this.isChannelsLast}_${this.sequentialAccessByThreads}`}getUserCode(){let e=this.isVec4?La(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner):Ba(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner,!1,null,this.sequentialAccessByThreads),t=this.isVec4?[this.innerElementSize,4,4]:[1,1,1];return`
    ${Oc(this.isChannelsLast,this.fitAOuter,this.fitBOuter,this.fitInner,this.addBias,this.activation,this.hasPreluActivationWeights,t[0],t[1],t[2])}
    ${e}
  `}},Ac=class{constructor(e,t=!1,n=null,r=!1){this.variableNames=[`x`,`W`],this.uniforms=`filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>,`,this.workgroupSize=[4,4,8],this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat===`channelsLast`,this.dispatchLayout=this.isChannelsLast?{x:[2],y:[1],z:[0,3]}:{x:[3],y:[2],z:[0,1]},this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t,this.activation=n,this.hasPreluActivationWeights=r,t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),this.shaderKey=`conv2dnaive_${this.activation}_${this.isChannelsLast}`}getUserCode(){return`
       ${K(this.activation,this.hasPreluActivationWeights,!1,4)}
       fn readInp(batch : i32, row : i32, col : i32, chan : i32) -> f32{
         let coords = vec4<i32>(batch, row, col, chan);
         if (coordsInBounds4D(coords, uniforms.xShape)) {
           return  getX(batch, row, col, chan);
         } else {
          return 0.0;
         }
       }
       fn readFilt(row : i32, col : i32, xChannel : i32, outChannel : i32) -> f32{
         let coords = vec4<i32>(row, col, xChannel, outChannel);
         if(coordsInBounds4D(coords, uniforms.wShape)) {
           return getW(row, col, xChannel, outChannel);
          } else {
            return 0.0;
          }
       }
       fn writeResult(batch : i32, row : i32, col : i32, chan : i32, valueIn : f32) {
         let coords = ${this.isChannelsLast?`vec4<i32>(batch, row, col, chan);`:`vec4<i32>(batch, chan, row, col);`}
         if (coordsInBounds4D(coords, uniforms.outShape)) {
           var value = valueIn;
           ${Ma(this.addBias,this.activation)}
           setOutputAtCoords(coords.x, coords.y, coords.z, coords.w, value);
         }
       }
       ${z(`index`)} {
         let coords = getOutputCoords();
         let batch = coords[0];
         let outChannel = ${this.isChannelsLast?`coords[3];`:`coords[1];`}
         let outRow = ${this.isChannelsLast?`coords[1];`:`coords[2];`}
         let outCol = ${this.isChannelsLast?`coords[2];`:`coords[3];`}
         var acc : f32 = 0.0;
         for (var row = 0; row < uniforms.filterDims[0]; row = row + 1) {
           for (var col = 0; col < uniforms.filterDims[1]; col = col + 1) {
             let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * row - uniforms.pads[0];
             let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * col - uniforms.pads[1];
             for (var xChannel = 0; xChannel < ${this.isChannelsLast?`uniforms.xShape[3];`:`uniforms.xShape[1];`} xChannel = xChannel + 1) {
               ${this.isChannelsLast?`let v = readInp(batch, xRow, xCol, xChannel);`:`let v = readInp(batch, xChannel, xRow, xCol);`}
               let f = readFilt(row, col, xChannel, outChannel);
               acc = acc + v * f;
             }
           }
         }
         writeResult(batch, outRow, outCol, outChannel, acc);
       }
     `}},jc=class{constructor(e,t){this.variableNames=[`x`],this.uniforms=`pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.isChannelsLast=t,this.shaderKey=`im2col_${this.isChannelsLast}`}getUserCode(){let e=this.isChannelsLast?1:2,t=this.isChannelsLast?2:3,n=this.isChannelsLast?`coords[1]`:`coords[2]`,r=this.isChannelsLast?`coords[2]`:`coords[1]`,i=this.isChannelsLast?`getX(batch, xRow, xCol, ch)`:`getX(batch, ch, xRow, xCol)`;return`
    ${z(`index`)} {
      let coords = getCoordsFromIndex(index);
      if(index < uniforms.size) {
        let batch = coords[0];
        let row = ${n};
        let col = ${r};
        let offsetY = (row / uniforms.outWidth) * uniforms.strides[0] - uniforms.pads[0];
        let xRow = offsetY + uniforms.dilations[0] * (col / uniforms.itemsPerBlockRow);
        var value = 0.0;
        if(xRow < uniforms.xShape[${e}] && xRow >= 0) {
          let offsetX = (row % uniforms.outWidth) * uniforms.strides[1] -
              uniforms.pads[1];
          let xCol = offsetX + uniforms.dilations[1] * ((col %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
          let ch = col % uniforms.inChannels;
          if(xCol < uniforms.xShape[${t}] && xCol >= 0) {
            value = ${i};
          }
        }
        setOutputAtIndex(index, value);
      }
    }
   `}};function Mc(e,t){let n=e.length;return n>=3?t?[...e.slice(0,-3),e[n-3]*e[n-2],e[n-1]]:[...e.slice(0,-3),e[n-3],e[n-2]*e[n-1]]:!t&&n===1&&e[0]>1?[e[0],1]:null}function Nc({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let c=n.dataFormat===`channelsLast`,l=!c,u=c&&n.filterHeight===n.inHeight&&n.filterWidth===n.inWidth&&n.padInfo.type===`VALID`,d=[],f,p;if(u){let i=n.inHeight*n.inWidth*n.inChannels;f=J({inputs:{x:e},backend:r,attrs:{shape:[1,n.batchSize,i]}}),p=J({inputs:{x:t},backend:r,attrs:{shape:[1,i,n.outChannels]}})}else f=J({inputs:{x:e},backend:r,attrs:{shape:c?[n.batchSize,n.inHeight*n.inWidth,n.inChannels]:[n.batchSize,n.inChannels,n.inHeight*n.inWidth]}}),p=J({inputs:{x:t},backend:r,attrs:{shape:[1,n.inChannels,n.outChannels]}});if(d.push(f),d.push(p),a!=null){let e=Mc(a.shape,c);e!=null&&(a=J({inputs:{x:a},backend:r,attrs:{shape:e}}),d.push(a))}if(i!=null){let e=Mc(i.shape,c);e!=null&&(i=J({inputs:{x:i},backend:r,attrs:{shape:e}}),d.push(i))}let m=$a({a:c?f:p,b:c?p:f,transposeA:l,transposeB:!1,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o}),h=J({inputs:{x:m},backend:r,attrs:{shape:n.outShape}});d.push(m);for(let e of d)r.disposeData(e.dataId);return h}function Pc({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let{filterWidth:c,filterHeight:l,inChannels:u,strideWidth:d,strideHeight:f,padInfo:p,outWidth:m,outHeight:h,dilationWidth:g,dilationHeight:_,dataFormat:v}=n,y=v===`channelsLast`,b=c*l*u,x=h*m,S=new jc(y?[n.batchSize,x,b]:[n.batchSize,b,x],y),C=[{type:`int32`,data:[p.top,p.left]},{type:`int32`,data:[f,d]},{type:`int32`,data:[_,g]},{type:`int32`,data:[m]},{type:`int32`,data:[u*c]},{type:`int32`,data:[u]}],w=r.runWebGPUProgram(S,[e],e.dtype,C),T=[];T.push(w);let E=J({inputs:{x:t},backend:r,attrs:{shape:[1,b,-1]}});if(T.push(E),a!=null){let e=Mc(a.shape,y);e!=null&&(a=J({inputs:{x:a},backend:r,attrs:{shape:e}}),T.push(a))}if(i!=null){let e=Mc(i.shape,y);e!=null&&(i=J({inputs:{x:i},backend:r,attrs:{shape:e}}),T.push(i))}let ee=$a({a:y?w:E,b:y?E:w,transposeA:!y,transposeB:!1,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o}),D=J({inputs:{x:ee},backend:r,attrs:{shape:n.outShape}});T.push(ee);for(let e of T)r.disposeData(e.dataId);return D}function Fc({x:e,filter:t,convInfo:n,backend:r,bias:i=null,preluActivationWeights:a=null,leakyreluAlpha:o=0,activation:s=null}){let c=i!=null,l=a!=null,u=n.dataFormat===`channelsLast`,d=u&&n.filterHeight===n.inHeight&&n.filterWidth===n.inWidth&&n.padInfo.type===`VALID`,f=j().getBool(`WEBGPU_USE_NAIVE_CONV2D_DEBUG`);if(!f&&(d||n.filterHeight===1&&n.filterWidth===1&&n.dilationHeight===1&&n.dilationWidth===1&&n.strideHeight===1&&n.strideWidth===1&&(n.padInfo.type===`SAME`||n.padInfo.type===`VALID`)))return Nc({x:e,filter:t,convInfo:n,backend:r,bias:i,activation:s,preluActivationWeights:a,leakyreluAlpha:o});let p=j().getNumber(`WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL`),m=p>-1?p:r.thresholdToIncreaseWorkgroups,h=n.batchSize*Math.ceil(n.outHeight*n.outWidth/32)*Math.ceil(n.outChannels/32);if(j().getBool(`WEBGPU_CONV_SEPARATE_IM2COL_SHADER`)||h<=m)return Pc({x:e,filter:t,convInfo:n,backend:r,bias:i,preluActivationWeights:a,leakyreluAlpha:o,activation:s});let g,_=[n.padInfo.top,n.padInfo.left],v=[{type:`int32`,data:[n.filterHeight,n.filterWidth]},{type:`int32`,data:[..._]},{type:`int32`,data:[n.strideHeight,n.strideWidth]},{type:`int32`,data:[n.dilationHeight,n.dilationWidth]}];if(f)g=new Ac(n,c,s,l);else{let e=u?n.outHeight*n.outWidth:n.outChannels,t=u?n.outChannels:n.outHeight*n.outWidth,i=n.filterHeight*n.filterWidth*n.inChannels;v.push({type:`int32`,data:[e]},{type:`int32`,data:[t]},{type:`int32`,data:[i]}),g=new kc(n,e,t,i,c,s,l,r.adapterInfo.isIntel())}let y=[],b=[e,t];c&&(!u&&i.shape.length===1&&(i=J({inputs:{x:i},backend:r,attrs:{shape:[i.shape[0],1,1]}}),y.push(i)),b.push(i)),l&&(!u&&a.shape.length===1&&(a=J({inputs:{x:a},backend:r,attrs:{shape:[a.shape[0],1,1]}}),y.push(a)),b.push(a)),s===`leakyrelu`&&(v.push({type:`float32`,data:[o]}),g.uniforms+=` alpha : f32,`);let x=r.runWebGPUProgram(g,b,e.dtype,v);for(let e of y)r.disposeData(e.dataId);return x}function Ic(e){let{inputs:t,attrs:n,backend:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=n,d=Vt(c);return Fc({x:i,filter:a,convInfo:A(i.shape,a.shape,o,l,s,u,!1,d),backend:r})}var Lc={kernelName:Sr,backendName:`webgpu`,kernelFunc:Ic},Rc=class{constructor(e){this.variableNames=[`dy`,`W`],this.uniforms=`filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, outBackprop : vec4<i32>,`,this.workgroupSize=[64,1,1],this.size=!1,this.isVec4=!1,this.workPerThread=1,this.outputShape=e.inShape,this.isChannelsLast=e.dataFormat===`channelsLast`,this.isVec4=this.isChannelsLast&&e.outChannels%4==0&&e.inChannels%4==0,this.isVec4?(this.workPerThread=2,this.outputComponent=4,this.workgroupSize=[4,4,4],this.dispatchLayout={x:[3],y:[2],z:[0,1]},this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[4,this.workPerThread,1])):(this.size=!0,this.workPerThread=1,this.workgroupSize=[64,1,1],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize)),this.shaderKey=`conv2DDerInput_${this.isChannelsLast}_${this.isVec4}_${this.workPerThread}`}getUserCode(){let e=this.isChannelsLast?1:2,t=this.isChannelsLast?2:3,n=this.isChannelsLast?3:1,r=`
    ${z()} {
      let batch = i32(globalId.z) / uniforms.outShape[1];
      let r = i32(globalId.z) % uniforms.outShape[1];
      let c = i32(globalId.y) * ${this.workPerThread};
      let d1 = i32(globalId.x) * 4;

      let dyCorner = vec2<i32>(r, c) - uniforms.pads;

      // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
      // ? = to be determined. : = across all values in that axis.
      var dotProd: array<vec4<f32>, ${this.workPerThread}>;
      for (var i = 0; i < ${this.workPerThread}; i++) {
        dotProd[i] = vec4<f32>(0.0);
      }
      for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + 1) {
        let dyR = f32(dyCorner.x + wR) / f32(uniforms.strides.x);
        let wRPerm = uniforms.filterDims.x - 1 - wR;
        if (dyR < 0.0 || dyR >= f32(uniforms.outBackprop[1]) ||
            fract(dyR) > 0.0) {
          continue;
        }
        let idyR = i32(dyR);

        for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + 1) {
          let dyC = f32(dyCorner.y + wC) / f32(uniforms.strides.y);
          let dyC2 = f32(dyCorner.y + 1 + wC) / f32(uniforms.strides.y);
          let wCPerm = uniforms.filterDims.y - 1 - wC;
          var bDyCVal = true;
          var bDyCVal2 = true;
          if (dyC < 0.0 || dyC >= f32(uniforms.outBackprop[2]) ||
              fract(dyC) > 0.0) {
            bDyCVal = false;
          }
          if (dyC2 < 0.0 || dyC2 >= f32(uniforms.outBackprop[2]) ||
              fract(dyC2) > 0.0) {
            bDyCVal2 = false;
          }

          let idyC = i32(dyC);
          let idyC2 = i32(dyC2);
          if (bDyCVal && bDyCVal2) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[0] = dotProd[0] + tmpval;
              xValue = getDy(batch, idyR, idyC2, d2);
              dotProd[1] = dotProd[1] + vec4<f32>(dot(xValue, wValue0),
                                                  dot(xValue, wValue1),
                                                  dot(xValue, wValue2),
                                                  dot(xValue, wValue3));
            }
          } else if (bDyCVal) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[0] = dotProd[0] + tmpval;
            }
          } else if (bDyCVal2) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC2, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[1] = dotProd[1] + tmpval;
            }
          }
        }
      }

      for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
        let coords = vec4<i32>(batch, r, c + i, d1);
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
        }
      }
    }
    `;return this.isVec4?`
    ${r}
    `:`
    ${z(`index`)} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[${n}];

        let dyCorner = vec2<i32>(coords[${e}], coords[${t}]) - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + 1) {
          let dyR = (f32(dyRCorner) + f32(wR)) / f32(uniforms.strides.x);
          let wRPerm = uniforms.filterDims.x - 1 - wR;
          if (dyR < 0.0 || dyR >= f32(uniforms.outBackprop[1]) || fract(dyR) > 0.0 ||
              wRPerm < 0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + 1) {
            let dyC = (f32(dyCCorner) + f32(wC)) / f32(uniforms.strides.y);
            let wCPerm = uniforms.filterDims.y - 1 - wC;
            if (dyC < 0.0 || dyC >= f32(uniforms.outBackprop[2]) ||
                fract(dyC) > 0.0 || wCPerm < 0) {
              continue;
            }
            let idyC = i32(dyC);

            for (var d2 = 0; d2 < uniforms.outBackprop[3]; d2 = d2 + 1) {
              let xValue = ${this.isChannelsLast?`getDy(batch, idyR, idyC, d2)`:`getDy(batch, d2, idyR, idyC)`};
              let wValue = getW(wRPerm, wCPerm, d1, d2);
              dotProd = dotProd + xValue * wValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `}},zc=class{constructor(e){this.variableNames=[`x`,`dy`],this.uniforms=`pads : vec2<i32>, strides : vec2<i32>, batchSize : i32, outHeight : i32, outWidth : i32, inHeight : i32, inWidth : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.filterShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.isChannelsLast=e.dataFormat===`channelsLast`,this.shaderKey=`conv2DDerFilter_${this.isChannelsLast}`}getUserCode(){return`
    ${z(`index`)} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let d2 = coords[3];

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b = b + 1) {
          for (var yR = 0; yR < uniforms.outHeight; yR = yR + 1) {
            let xR = wR + yR * uniforms.strides[0] - uniforms.pads[0];
            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC = yC + 1) {
              let xC = wC + yC * uniforms.strides[1] - uniforms.pads[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              if (${this.isChannelsLast}) {
                let dyValue = getDy(b, yR, yC, d2);
                let xValue = getX(b, xR, xC, d1);
                dotProd = dotProd + xValue * dyValue;
              } else {
                let dyValue = getDy(b, d2, yR, yC);
                let xValue = getX(b, d1, xR, xC);
                dotProd = dotProd + xValue * dyValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `}},Bc=class{constructor(e){this.variableNames=[`x`,`dy`],this.uniforms=`pads : vec3<i32>, strides : vec3<i32>, batchSize : i32, outDepth : i32,
       outHeight : i32, outWidth : i32, inDepth : i32, inHeight : i32, inWidth : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.filterShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`conv3DDerFilter`}getUserCode(){return`
    ${z(`index`)} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wF = coords.x;
        let wR = coords.y;
        let wC = coords.z;
        let d1 = coords.w;
        let d2 = coords.u;

        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b++) {
          for (var yF = 0; yF < uniforms.outDepth; yF++) {
            let xF = wF + yF * uniforms.strides[0] - uniforms.pads[0];
            if (xF < 0 || xF >= uniforms.inDepth) {
              continue;
            }

            for (var yR = 0; yR < uniforms.outHeight; yR++) {
              let xR = wR + yR * uniforms.strides[1] - uniforms.pads[1];
              if (xR < 0 || xR >= uniforms.inHeight) {
                continue;
              }

              for (var yC = 0; yC < uniforms.outWidth; yC++) {
                let xC = wC + yC * uniforms.strides[2] - uniforms.pads[2];
                if (xC < 0 || xC >= uniforms.inWidth) {
                  continue;
                }

                let dyValue = getDy(b, yF, yR, yC, d2);
                let xValue = getX(b, xF, xR, xC, d1);
                dotProd += xValue * dyValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `}},Vc=class{constructor(e){this.variableNames=[`dy`,`W`],this.uniforms=`filterDims : vec3<i32>, pads : vec3<i32>, strides : vec3<i32>,
      outDepth : i32, outHeight : i32, outWidth : i32, outChannels : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`conv3DDerInput`}getUserCode(){return`
    ${z(`index`)} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let d1 = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyFCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        var dotProd = 0.0;
        for (var wF = 0; wF < uniforms.filterDims[0]; wF++) {
          let dyF = f32(dyFCorner + wF) / f32(uniforms.strides[0]);
          if (dyF < 0.0 || dyF >= f32(uniforms.outDepth) || fract(dyF) > 0.0) {
            continue;
          }
          let idyF = i32(dyF);

          let wFPerm = uniforms.filterDims[0] - 1 - wF;

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            let wRPerm = uniforms.filterDims[1] - 1 - wR;

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let wCPerm = uniforms.filterDims[2] - 1 - wC;

              for (var d2 = 0; d2 < uniforms.outChannels; d2++) {
                let xValue = getDy(batch, idyF, idyR, idyC, d2);
                let wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `}};function Hc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,dataFormat:c,dimRoundingMode:l,filterShape:u}=r,d=Vt(c),f=A(i.shape,u,o,1,s,l,!1,d),p=new zc(f),m=[{type:`int32`,data:[f.padInfo.top,f.padInfo.left]},{type:`int32`,data:[f.strideHeight,f.strideWidth]},{type:`int32`,data:[f.batchSize]},{type:`int32`,data:[f.outHeight]},{type:`int32`,data:[f.outWidth]},{type:`int32`,data:[f.inHeight]},{type:`int32`,data:[f.inWidth]}];return n.runWebGPUProgram(p,[i,a],i.dtype,m)}var Uc={kernelName:u,backendName:`webgpu`,kernelFunc:Hc};function Wc(e=4){let t=e=>{switch(e){case 1:return`return W[getIndexFromCoords4D(coord, uniforms.wShape)];`;case 4:return`
            let coord1 = vec4<i32>(coordX, coordY, col + 1, rowInner);
            let coord2 = vec4<i32>(coordX, coordY, col + 2, rowInner);
            let coord3 = vec4<i32>(coordX, coordY, col + 3, rowInner);
            let v0 = W[getIndexFromCoords4D(coord, uniforms.wShape)];
            let v1 = W[getIndexFromCoords4D(coord1, uniforms.wShape)];
            let v2 = W[getIndexFromCoords4D(coord2, uniforms.wShape)];
            let v3 = W[getIndexFromCoords4D(coord3, uniforms.wShape)];
            return vec4<f32>(v0, v1, v2, v3);
            `;default:throw Error(`innerElementSize ${e} is not supported.`)}},n=`if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${`
      let outRow = row / uniforms.outShape[2];
      let outCol = row % uniforms.outShape[2];

      let WRow = col / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
      let WCol = col / uniforms.outBackprop[3] % uniforms.filterDims[1];
      let xR = f32(outRow - uniforms.pads[0] + WRow) / f32(uniforms.strides[0]);
      let xC = f32(outCol - uniforms.pads[1] + WCol) / f32(uniforms.strides[1]);
      if (xR < 0.0 || xR >= f32(uniforms.outBackprop[1]) || fract(xR) > 0.0) {
        return ${I(e)}(0.0);
      }
      if (xC < 0.0 || xC >= f32(uniforms.outBackprop[2]) || fract(xC) > 0.0) {
        return ${I(e)}(0.0);
      }
      let coord = vec4<i32>(
          batch,
          i32(xR),
          i32(xC),
          col % uniforms.outBackprop[3]);
      return x[getIndexFromCoords4D(coord, uniforms.xShape)/${e}];`}
      }
      return ${I(e)}(0.0);`;return`
  fn mm_readA(batch: i32, row : i32, col : i32) -> ${I(e)} {
    ${n}
  }

  fn mm_readB(batch: i32, row : i32, col : i32) -> ${I(e)} {
    let coordX = uniforms.filterDims.x - 1 -
        row / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
    let coordY = uniforms.filterDims.y - 1 -
        (row / uniforms.outBackprop[3]) % uniforms.filterDims[1];
    if (row < uniforms.dimInner && col < uniforms.dimBOuter &&
        coordX >= 0 && coordY >= 0) {
      let rowInner = row % uniforms.outBackprop[3];
      let coord = vec4<i32>(coordX, coordY, col, rowInner);
      ${t(e)}
    }
    return ${I(e)}(0.0);
  }

  fn mm_write(batch: i32, row : i32, col : i32, valueInput : ${I(e)}) {
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
      var value = valueInput;
      let outCoord = vec4<i32>(
          batch,
          row / uniforms.outShape[2],
          row % uniforms.outShape[2],
          col);
      result[getIndexFromCoords4D(outCoord, uniforms.outShape)/${e}] = value;
    }
  }`}var Gc=class{constructor(e){this.variableNames=[`x`,`W`],this.uniforms=`filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, outBackprop : vec4<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`,this.outputShape=e.inShape,k(e.dataFormat===`channelsLast`,()=>`TODO: NCHW is unimplemented`),this.isVec4=e.inChannels%4==0&&e.outChannels%4==0,this.dispatchLayout={x:[3],y:[1,2],z:[0]},this.workgroupSize=ri(this.dispatchLayout,this.outputShape,this.isVec4),this.elementsPerThread=ii(this.dispatchLayout,this.outputShape,this.isVec4),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread),this.isVec4&&(this.outputComponent=4,this.variableComponents=[4,1]),this.shaderKey=`conv2DDerInputMM_${this.isVec4}_${this.elementsPerThread}`}getUserCode(){let e=this.isVec4?La(this.elementsPerThread,this.workgroupSize):Ba(this.elementsPerThread,this.workgroupSize);return`
    ${Wc(this.isVec4?4:1)}
    ${e}
    `}};function Kc(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{inputShape:o,strides:s,pad:c,dataFormat:l,dimRoundingMode:u}=r,d=Vt(l),f=A(o,a.shape,s,1,c,u,!1,d),p=[{type:`int32`,data:[f.filterHeight,f.filterWidth]},{type:`int32`,data:[f.filterHeight-1-f.padInfo.top,f.filterWidth-1-f.padInfo.left]},{type:`int32`,data:[f.strideHeight,f.strideWidth]},{type:`int32`,data:[f.batchSize,f.outHeight,f.outWidth,f.outChannels]}],m;if(j().getBool(`WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE`)||f.dataFormat!==`channelsLast`)m=new Rc(f);else{m=new Gc(f);let e=f.inHeight*f.inWidth,t=f.inChannels,n=f.filterHeight*f.filterWidth*f.outChannels;p.push({type:`uint32`,data:[e]},{type:`uint32`,data:[t]},{type:`uint32`,data:[n]})}return n.runWebGPUProgram(m,[i,a],`float32`,p)}var qc={kernelName:st,backendName:`webgpu`,kernelFunc:Kc},Jc=class{constructor(e){this.variableNames=[`x`,`W`],this.uniforms=`filterDims: vec3<i32>, pads: vec3<i32>, strides: vec3<i32>, dilations: vec3<i32>,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`conv3dnaive`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords.x;
        let d2 = coords.u;

        let xFRCCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
        let xFCorner = xFRCCorner.x;
        let xRCorner = xFRCCorner.y;
        let xCCorner = xFRCCorner.z;

        let inputDepthNearestVec4 = (uniforms.xShape.u / 4) * 4;
        let inputDepthVec4Remainder = uniforms.xShape.u % 4;

        var dotProd = 0.0;
        for (var wF = 0; wF < uniforms.filterDims[0]; wF++) {
          let xF = xFCorner + wF * uniforms.dilations[0];
          if (xF < 0 || xF >= uniforms.xShape.y) {
            continue;
          }

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let xR = xRCorner + wR * uniforms.dilations[1];
            if (xR < 0 || xR >= uniforms.xShape.z) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let xC = xCCorner + wC * uniforms.dilations[2];
              if (xC < 0 || xC >= uniforms.xShape.w) {
                continue;
              }

              for (var d1 = 0; d1 < inputDepthNearestVec4; d1 += 4) {
                let xValues = vec4<f32>(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                let wValues = vec4<f32>(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (inputDepthVec4Remainder == 1) {
                dotProd += getX(batch, xF, xR, xC, inputDepthNearestVec4) *
                  getW(wF, wR, wC, inputDepthNearestVec4, d2);
              } else if (inputDepthVec4Remainder == 2) {
                let xValues = vec2<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1)
                );
                let wValues = vec2<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (inputDepthVec4Remainder == 3) {
                let xValues = vec3<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 2)
                );
                let wValues = vec3<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }`}};function Yc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=cn(i.shape,a.shape,o,c,s),u=[l.padInfo.front,l.padInfo.top,l.padInfo.left],d=[{type:`int32`,data:[l.filterDepth,l.filterHeight,l.filterWidth]},{type:`int32`,data:[...u]},{type:`int32`,data:[l.strideDepth,l.strideHeight,l.strideWidth]},{type:`int32`,data:[l.dilationDepth,l.dilationHeight,l.dilationWidth]}],p=new Jc(l),m=f(i.dtype,a.dtype);return n.runWebGPUProgram(p,[i,a],m,d)}var Xc={kernelName:se,backendName:`webgpu`,kernelFunc:Yc};function Zc(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,pad:s,filterShape:c}=r,l=cn(i.shape,c,o,1,s),u=new Bc(l),d=[{type:`int32`,data:[l.padInfo.front,l.padInfo.top,l.padInfo.left]},{type:`int32`,data:[l.strideDepth,l.strideHeight,l.strideWidth]},{type:`int32`,data:[l.batchSize]},{type:`int32`,data:[l.outDepth]},{type:`int32`,data:[l.outHeight]},{type:`int32`,data:[l.outWidth]},{type:`int32`,data:[l.inDepth]},{type:`int32`,data:[l.inHeight]},{type:`int32`,data:[l.inWidth]}];return n.runWebGPUProgram(u,[i,a],a.dtype,d)}var Qc={kernelName:rt,backendName:`webgpu`,kernelFunc:Zc};function $c(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,pad:s,inputShape:c}=r,l=cn(c,a.shape,o,1,s),u=new Vc(l),d=[{type:`int32`,data:[l.filterDepth,l.filterHeight,l.filterWidth]},{type:`int32`,data:[l.filterDepth-1-l.padInfo.front,l.filterHeight-1-l.padInfo.top,l.filterWidth-1-l.padInfo.left]},{type:`int32`,data:[l.strideDepth,l.strideHeight,l.strideWidth]},{type:`int32`,data:[l.outDepth]},{type:`int32`,data:[l.outHeight]},{type:`int32`,data:[l.outWidth]},{type:`int32`,data:[l.outChannels]}];return n.runWebGPUProgram(u,[i,a],i.dtype,d)}var el={kernelName:dt,backendName:`webgpu`,kernelFunc:$c},tl={kernelName:`Cos`,backendName:`webgpu`,kernelFunc:X({opType:W.COS})},nl=X({opType:W.COSH}),rl={kernelName:ve,backendName:`webgpu`,kernelFunc:nl},il=class{constructor(e,t,n,r){this.variableNames=[`Image`,`Boxes`,`BoxInd`],this.uniforms=`extrapolationValue : f32,`,this.workgroupSize=[64,1,1],this.size=!0;let[i]=t;this.outputShape=[i,n[0],n[1],e],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.methodId=+(r===`bilinear`),this.cropHeightBiggerThan1=this.outputShape[1]>1,this.cropWidthBiggerThan1=this.outputShape[2]>1,this.shaderKey=`cropAndResize_${this.methodId}_${this.cropHeightBiggerThan1}_${this.cropWidthBiggerThan1}`}getUserCode(){let[e,t]=[`f32(uniforms.imageShape[1] - 1)`,`f32(uniforms.imageShape[2] - 1)`],[n,r,i]=this.cropHeightBiggerThan1?[`(${e} / f32(uniforms.outShape[1] - 1))`,`(y2-y1) * height_ratio`,`y1*${e} + f32(y)*(height_scale)`]:[`0.0`,`0.0`,`0.5 * (y1+y2) * ${e}`],[a,o,s]=this.cropWidthBiggerThan1?[`(${t} / f32(uniforms.outShape[2] - 1))`,`(x2-x1) * width_ratio`,`x1*${t} + f32(x)*(width_scale)`]:[`0.0`,`0.0`,`0.5 * (x1+x2) * ${t}`];return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let height_ratio = f32(${n});
        let width_ratio = f32(${a});
        let b = coords[0];
        let y = coords[1];
        let x = coords[2];
        let d = coords[3];
        // get box vals
        let y1 = getBoxes(b, 0);
        let x1 = getBoxes(b, 1);
        let y2 = getBoxes(b, 2);
        let x2 = getBoxes(b, 3);
        // get image in batch index
        let bInd = i32(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= uniforms.outShape[0]) {
          return;
        }
        let height_scale = ${r};
        let width_scale = ${o};
        let in_y = ${i};
        if( in_y < 0.0 || in_y > ${e} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let in_x = ${s};
        if( in_x < 0.0 || in_x > ${t} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let sourceFracIndexCR = vec2<f32>(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          let sourceFloorCR = vec2<i32>(sourceFracIndexCR);
          let sourceCeilCR = vec2<i32>(ceil(sourceFracIndexCR));
          let topLeft = getImage(bInd, sourceFloorCR.y, sourceFloorCR.x, d);
          let bottomLeft = getImage(bInd, sourceCeilCR.y, sourceFloorCR.x, d);
          let topRight = getImage(bInd, sourceFloorCR.y, sourceCeilCR.x, d);
          let bottomRight = getImage(bInd, sourceCeilCR.y, sourceCeilCR.x, d);
          let fracCR = sourceFracIndexCR - vec2<f32>(sourceFloorCR);
          let top = topLeft + (topRight - topLeft) * fracCR.x;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          let newValue = top + (bottom - top) * fracCR.y;
          setOutputAtIndex(index, newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          let sourceNearestCR = vec2<i32>(floor(
            sourceFracIndexCR + vec2<f32>(0.5,0.5)));
          let newValue = getImage(
            bInd, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutputAtIndex(index, newValue);
        }
      }
    }
    `}},al={kernelName:Vn,backendName:`webgpu`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{image:i,boxes:a,boxInd:o}=t,{cropSize:s,method:c,extrapolationValue:l}=r,u=new il(i.shape[3],a.shape,s,c),d=[{type:`float32`,data:[l]}];return n.runWebGPUProgram(u,[i,a,o],`float32`,d)}},ol;(function(e){e.Prod=`*`,e.Sum=`+`})(ol||={});var sl=class{constructor(e,t,n,r){this.variableNames=[`x`],this.uniforms=`index : f32,`,this.size=!0,this.workgroupSize=[128,1,1],this.outputShape=t,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.exclusive=n,this.reverse=r,this.op=e,this.shaderKey=`cum_${this.op}_${this.exclusive}_${this.reverse}`}getUserCode(){let e=this.outputShape.length,t=this.op===ol.Prod?`1.0`:`0.0`,n=this.exclusive?t:`getX(${cl(e,`coords`,this.op)})`,r=this.outputShape[this.outputShape.length-1],i=``,a=``;return this.exclusive?(i=this.reverse?`end != ${r-1}`:`end != 0`,a=this.reverse?`end + 1`:`end - 1`):(i=this.reverse?`end + pow2 < ${r}`:`end >= pow2`,a=this.reverse?`end + pow2`:`end - pow2`),`
      ${z(`index`)} {
       if (index < uniforms.size) {
         var coords = getCoordsFromIndex(index);

         let end = ${ll(e,`coords`,this.op)};
         var val = ${n};
         let pow2 = i32(pow(2.0, uniforms.index));
         if (${i}) {
           let idx = ${a};
           ${ll(e,`coords`,this.op)} = idx;
           val ${this.op}= getX(${cl(e,`coords`,this.op)});
         }
         setOutputAtIndex(index, val);
       }
      }
    `}};function cl(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.x, ${t}.y`;if(e===3)return`${t}.x, ${t}.y, ${t}.z`;if(e===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function ll(e,t,n){if(e===1)return`${t}`;if(e===2)return`${t}.y`;if(e===3)return`${t}.z`;if(e===4)return`${t}.w`;throw Error(`Cumulative ${n} for rank ${e} is not yet supported`)}function ul(e,t,n,r,i,a){let o=t.shape.length,s=nr([r],o),c=t;s!=null&&(c=Q({inputs:{x:t},backend:n,attrs:{perm:s}}));let l=Nn(1,o)[0];if(l!==o-1)throw Error(`WebGPU cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${r}`);let u=c.shape[l],d=Y({inputs:{x:c},backend:n});for(let t=0;t<=Math.ceil(Math.log2(u))-1;t++){let r=new sl(e,c.shape,!1,a),i=d,o=[{type:`float32`,data:[t]}];d=n.runWebGPUProgram(r,[d],d.dtype,o),n.disposeData(i.dataId)}if(i){let t=new sl(e,c.shape,i,a),r=d;d=n.runWebGPUProgram(t,[d],d.dtype,[{type:`float32`,data:[0]}]),n.disposeData(r.dataId)}if(s!=null){let e=qn(s),t=Q({inputs:{x:d},backend:n,attrs:{perm:e}});return n.disposeData(d.dataId),n.disposeData(c.dataId),t}return d}function dl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return ul(ol.Prod,i,n,a,o,s)}var fl={kernelName:he,backendName:`webgpu`,kernelFunc:dl};function pl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,exclusive:o,reverse:s}=r;return ul(ol.Sum,i,n,a,o,s)}var ml={kernelName:xt,backendName:`webgpu`,kernelFunc:pl};function hl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,weights:a}=t,{size:o,binaryOutput:s}=r,c=i.shape.length===1,l=M(a.shape)>0,u=a.dtype,d=c?[i.shape[0]]:[i.shape[0],i.shape[1]],f=q({backend:n,attrs:{shape:c?[o]:[i.shape[0],o],value:0,dtype:u}}),p=new Qs(d,l,s),m=[{type:`int32`,data:[o]}],h=l?[i,a]:[i];return n.runWebGPUProgram(p,h,u,m,f)}var gl={kernelName:gt,backendName:`webgpu`,kernelFunc:hl},_l=class{constructor(e,t){this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.uniforms=`blockSize : i32,`,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`depthToSpace_${t}`,this.dataFormat=t}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let h = ${this.getHeightCoordString()};
          let w = ${this.getWidthCoordString()};
          let d = ${this.getDepthCoordString()};

          let in_h = h / uniforms.blockSize;
          let offset_h = h % uniforms.blockSize;
          let in_w = w / uniforms.blockSize;
          let offset_w = w % uniforms.blockSize;
          let offset_d = (offset_h * uniforms.blockSize + offset_w) *
            ${this.getOutputDepthSize()};
          let in_d = d + offset_d;

          let rlt = ${this.getInputSamplingString()};
          setOutputAtIndex(index, rlt);
        }
      }`}getHeightCoordString(){return this.dataFormat===`NHWC`?`coords[1]`:`coords[2]`}getWidthCoordString(){return this.dataFormat===`NHWC`?`coords[2]`:`coords[3]`}getDepthCoordString(){return this.dataFormat===`NHWC`?`coords[3]`:`coords[1]`}getOutputDepthSize(){return this.dataFormat===`NHWC`?`uniforms.outShape[3]`:`uniforms.outShape[1]`}getInputSamplingString(){return this.dataFormat===`NHWC`?`getX(b, in_h, in_w, in_d)`:`getX(b, in_d, in_h, in_w)`}};function vl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockSize:a,dataFormat:o}=r,s=i.shape[0],c=o===`NHWC`?i.shape[1]:i.shape[2],l=o===`NHWC`?i.shape[2]:i.shape[3],u=o===`NHWC`?i.shape[3]:i.shape[1],d=c*a,f=l*a,p=u/(a*a),m=o===`NHWC`?[s,d,f,p]:[s,p,d,f],h=[{type:`int32`,data:[a]}],g=new _l(m,o);return n.runWebGPUProgram(g,[i],i.dtype,h)}var yl={kernelName:Dt,backendName:`webgpu`,kernelFunc:vl},bl=class{constructor(e,t,n,r=!1,i=null,a=!1){this.variableNames=[`x`,`W`],this.uniforms=`pads : vec2<i32>, inDims : vec2<i32>,`,this.workgroupSize=[16,16,1],this.outputShape=e,this.dispatchLayout={x:[3],y:[2],z:[0,1]},this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),r&&this.variableNames.push(`bias`),a&&this.variableNames.push(`preluActivationWeights`),this.addBias=r,this.activation=i,this.hasPreluActivation=a,this.filterHeight=t,this.filterWidth=n,this.shaderKey=`depthwiseNCHW_${this.activation}_${this.filterHeight}_${this.filterWidth}`}getUserCode(){let e=this.filterWidth*this.filterHeight,t=this.workgroupSize[0]*this.workgroupSize[1]*this.workgroupSize[2],n=this.workgroupSize[1]+this.filterHeight-1,r=this.workgroupSize[0]+this.filterWidth-1;return`
      ${K(this.activation,this.hasPreluActivation,!1,4)}

      var<workgroup> mm_Asub : array<array<f32, ${r}>, ${n}>;
      var<workgroup> mm_Bsub : array<array<f32, ${this.filterWidth}>, ${this.filterHeight}>;
      fn readX(batch : i32, channel : i32, row : i32, col : i32) -> f32 {
        var value = 0.0;
        if (row >=0 && row < uniforms.inDims[0] && col >=0 && col < uniforms.inDims[1])
        {
          value = getX(batch, channel, row, col);
        }
        return value;
      }

      ${z()} {
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.zw) - uniforms.pads;
        let channelMul = uniforms.wShape[3];
        let d1 = coords[1] / channelMul;
        let q = coords[1] % channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;

        let localRow = i32(localId.y);
        let localCol = i32(localId.x);

        // Load one tile of X into local memory.
        for (var inputRow = localRow; inputRow < ${n}; inputRow = inputRow + ${this.workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${r}; inputCol = inputCol + ${this.workgroupSize[0]}) {
            let rowOffset = inputRow - localRow;
            let colOffset = inputCol - localCol;
            mm_Asub[inputRow][inputCol] = readX(batch, d1, inputRowStart + rowOffset, inputColStart + colOffset);
          }
        }

        // Load one tile of W into local memory.
        var wIndex = i32(localIndex);
        ${e<t?`if (wIndex < ${e})`:`for(; wIndex < ${e}; wIndex = wIndex + ${t})`}

        {
          let wRow = wIndex / ${this.filterWidth};
          let wCol = wIndex % ${this.filterWidth};
          mm_Bsub[wRow][wCol] = getW(wRow, wCol, d1, q);
        }

        workgroupBarrier();

        var value = 0.0;
        for (var wR = 0; wR < ${this.filterHeight}; wR = wR + 1) {
          for (var wC = 0; wC < ${this.filterWidth}; wC = wC + 1) {
            let xVal = mm_Asub[localRow + wR][localCol + wC];
            let wVal = mm_Bsub[wR][wC];
            value = fma(xVal, wVal, value);
          }
        }
        ${Ma(this.addBias,this.activation)}
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `}},xl=class{constructor(e,t=!1,n=null,r=!1){this.variableNames=[`x`,`W`],this.uniforms=`pads : vec2<i32>, inDims : vec2<i32>, virtualWidth : i32,`,this.workgroupSize=[64,1,1],this.workPerThread=4,this.outputComponent=4,this.outputShape=e.outShape,this.virtualWidth=Math.ceil(this.outputShape[2]/this.workPerThread)*this.workPerThread;let i=[this.outputShape[0],this.outputShape[1],this.virtualWidth,this.outputShape[3]];this.dispatchLayout=V(i),this.dispatch=B(this.dispatchLayout,i,this.workgroupSize,[this.outputComponent*this.workPerThread,1,1]),k(e.dataFormat===`channelsLast`,()=>`TODO: NCHW is unimplemented`),t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),this.convInfo=e,this.addBias=t,this.activation=n,this.hasPreluActivation=r,this.shaderKey=`depthwiseVec4_${n}_${this.convInfo.filterHeight}_${this.convInfo.filterWidth}_${this.convInfo.strideHeight}_${this.convInfo.strideWidth}_${this.workPerThread}`}getUserCode(){let e=(this.workPerThread-1)*this.convInfo.strideWidth+this.convInfo.filterWidth,t=this.convInfo.strideHeight,n=this.convInfo.strideWidth;return`
      ${K(this.activation,this.hasPreluActivation,!0,4)}
      fn readX(batch : i32, row : i32, col : i32, channel : i32) -> vec4<f32> {
        var value = vec4<f32>(0.0);
        if (col >=0 && col < uniforms.inDims[1]) {
          value = getX(batch, row, col, channel);
        }
        return value;
      }

      ${z(`index`)} {
        let width0 = uniforms.outShape[3] / ${this.outputComponent};
        let d1 = (index % width0) * ${this.outputComponent};
        var index1 = index / width0;
        let width1 = uniforms.virtualWidth / ${this.workPerThread};
        let c = (index1 % width1) * ${this.workPerThread};
        index1 = index1 / width1;
        let r = index1 % uniforms.outShape[1];
        let batch = index1 / uniforms.outShape[1];

        let xRCCorner = vec2<i32>(r, c) * vec2<i32>(${t}, ${n}) - uniforms.pads;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;
        var xVals : array<vec4<f32>, ${e}>;
        var dotProd : array<vec4<f32>, ${this.workPerThread}>;
        for (var i = 0; i < ${this.workPerThread}; i++) {
          dotProd[i] = vec4<f32>(0.0);
        }

        // Use constant instead of uniform can give better performance.
        for (var wR = 0; wR < ${this.convInfo.filterHeight}; wR = wR + 1) {
          let xR = xRCorner + wR;
          if (xR >=0 && xR < uniforms.inDims[0]) {
            for (var i = 0; i < ${e}; i++) {
              xVals[i] = readX(batch, xR, xCCorner + i, d1);
            }
            for (var wC = 0; wC < ${this.convInfo.filterWidth}; wC = wC + 1) {
              let wValue = getW(wR, wC, d1, 0);
              for (var i = 0; i < ${this.workPerThread}; i++) {
                dotProd[i] = fma(xVals[i * ${n} + wC], wValue, dotProd[i]);
              }
            }
          }
        }

        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let coords = vec4<i32>(batch, r, c + i, d1);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
            var value = dotProd[i];
            ${Ma(this.addBias,this.activation)}
            setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
          }
        }
      }
    `}},Sl=class{constructor(e,t=!1,n=null,r=!1){this.variableNames=[`x`,`W`],this.uniforms=`pads : vec2<i32>, inDims : vec2<i32>, filterHeight : i32,
      filterWidth : i32, strides : vec2<i32>, dilations : vec2<i32>,`,this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.isChannelsLast=e.dataFormat===`channelsLast`,t&&this.variableNames.push(`bias`),r&&this.variableNames.push(`preluActivationWeights`),this.convInfo=e,this.addBias=t,this.activation=n,this.hasPreluActivation=r,this.shaderKey=`depthwise_${this.activation}_${this.isChannelsLast}`}getUserCode(){let e=this.isChannelsLast?`getX(batch, xR, xC, d1);`:`getX(batch, d1, xR, xC);`;return`
      ${K(this.activation,this.hasPreluActivation,!1,4)}

      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let batch = coords[0];
          let xRCCorner = vec2<i32>(coords.${this.isChannelsLast?`yz`:`zw`}) * uniforms.strides - uniforms.pads;
          let d2 = coords[${this.isChannelsLast?3:1}];
          let channelMul = uniforms.wShape[3];
          let d1 = d2 / channelMul;
          let q = d2 % channelMul;

          let inputRowStart = xRCCorner.x;
          let inputColStart = xRCCorner.y;
          let inputRowEnd = inputRowStart + uniforms.filterHeight *
              uniforms.dilations[0];
          let inputColEnd = inputColStart + uniforms.filterWidth *
              uniforms.dilations[1];

          // Convolve x(?, ?, d1)|x(d1, ?, ?) with w(:, :, d1, q) to get
          // y(yR, yC, d2)|y(d2, yR, yC). ? = to be determined. : = across all
          // values in that axis. x(?, ?, d1) and y(yR, yC, d2) is for NHWC.
          // x(d1, ?, ?) and y(d2, yR, yC) is for NCHW.
          var value = 0.0;

          // Extract if checking out of for loop for performance.
          if (inputRowStart >= 0 && inputColStart >= 0 &&
            inputRowEnd < uniforms.inDims[0] &&
                inputColEnd < uniforms.inDims[1]) {
              for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
                let xR = inputRowStart + wR * uniforms.dilations[0];

                for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                  let xC = inputColStart + wC * uniforms.dilations[1];

                  let xVal = ${e};
                  let wVal = getW(wR, wC, d1, q);
                  value = value + xVal * wVal;
                }
              }
            } else {
              for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
                let xR = inputRowStart + wR * uniforms.dilations[0];

                if (xR < 0 || xR >= uniforms.inDims[0]) {
                  continue;
                }

                for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                  let xC = inputColStart + wC * uniforms.dilations[1];

                  if (xC < 0 || xC >= uniforms.inDims[1]) {
                    continue;
                  }

                  let xVal = ${e};
                  let wVal = getW(wR, wC, d1, q);
                  value = value + xVal * wVal;
                }
              }
            }
            ${Ma(this.addBias,this.activation)}
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `}};function Cl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dataFormat:c,dilations:l,dimRoundingMode:u}=r,d=Vt(c),f=l;f??=[1,1];let p=A(i.shape,a.shape,o,f,s,u,!0,d),m=[{type:`int32`,data:[p.padInfo.top,p.padInfo.left]},{type:`int32`,data:[p.inHeight,p.inWidth]}],h=p.dataFormat===`channelsLast`,g;return!h&&p.inHeight>16&&p.inWidth>16&&p.strideHeight===1&&p.strideWidth===1&&p.dilationWidth===1&&p.dilationHeight===1&&p.inChannels===p.outChannels?g=new bl(p.outShape,p.filterHeight,p.filterWidth):h&&p.outHeight>4&&p.outWidth>4&&p.strideWidth<=2&&p.inChannels===p.outChannels&&p.dilationHeight===1&&p.dilationWidth===1&&p.inChannels%4==0?(g=new xl(p),m.push({type:`int32`,data:[g.virtualWidth]})):(g=new Sl(p),m.push({type:`int32`,data:[p.filterHeight]},{type:`int32`,data:[p.filterWidth]},{type:`int32`,data:[p.strideHeight,p.strideWidth]},{type:`int32`,data:[p.dilationHeight,p.dilationWidth]})),n.runWebGPUProgram(g,[i,a],i.dtype,m)}var wl={kernelName:ze,backendName:`webgpu`,kernelFunc:Cl},Tl=class{constructor(e){this.variableNames=[`x`,`dy`],this.uniforms=`strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>, outHeight : i32,
      outWidth : i32, inHeight : i32, inWidth : i32, batchSize : i32, channelMul : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.filterShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`depthwise_conv2d_backprop_filter`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let dm = coords[3];
        let d2 = d1 * uniforms.channelMul + dm;

        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b++) {
          for (var yR = 0; yR < uniforms.outHeight; yR++) {
            let xR = wR + yR * uniforms.strides[0] - uniforms.pads[0];

            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC++) {
              let xC = wC + yC * uniforms.strides[1] - uniforms.pads[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              let dyValue = getDy(b, yR, yC, d2);
              let xValue = getX(b, xR, xC, d1);
              dotProd += xValue * dyValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `}},El=class{constructor(e){this.variableNames=[`dy`,`W`],this.uniforms=`strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, channelMul : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`depthwise_conv2d_backprop_input`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[3];
        let dyCorner = coords.yz - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }

          let idyR = i32(dyR);
          let wRPerm = uniforms.filterDims[0] - 1 - wR;

          for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }

            let idyC = i32(dyC);
            let wCPerm = uniforms.filterDims[1] - 1 - wC;

            for (var dm = 0; dm < uniforms.channelMul; dm++) {
              let d2 = d1 * uniforms.channelMul + dm;
              let xValue = getDy(batch, idyR, idyC, d2);
              let wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `}};function Dl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,dy:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,filterShape:u}=r,d=A(i.shape,u,o,s,c,l,!0),f=new Tl(d),p=[{type:`int32`,data:[d.strideHeight,d.strideWidth]},{type:`int32`,data:[d.padInfo.top,d.padInfo.left]},{type:`int32`,data:[d.filterHeight,d.filterWidth]},{type:`int32`,data:[d.outHeight]},{type:`int32`,data:[d.outWidth]},{type:`int32`,data:[d.inHeight]},{type:`int32`,data:[d.inWidth]},{type:`int32`,data:[d.batchSize]},{type:`int32`,data:[d.outChannels/d.inChannels]}];return n.runWebGPUProgram(f,[i,a],`float32`,p)}var Ol={kernelName:t,backendName:`webgpu`,kernelFunc:Dl};function kl(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,filter:a}=t,{strides:o,dilations:s,pad:c,dimRoundingMode:l,inputShape:u}=r,d=A(u,a.shape,o,s,c,l,!0),f=new El(d),p=[{type:`int32`,data:[d.strideHeight,d.strideWidth]},{type:`int32`,data:[d.filterHeight-1-d.padInfo.top,d.filterWidth-1-d.padInfo.left]},{type:`int32`,data:[d.filterHeight,d.filterWidth]},{type:`int32`,data:[d.outHeight]},{type:`int32`,data:[d.outWidth]},{type:`int32`,data:[d.outChannels/d.inChannels]}];return n.runWebGPUProgram(f,[i,a],i.dtype,p)}var Al={kernelName:Qt,backendName:`webgpu`,kernelFunc:kl},jl=class{constructor(e){this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e,e],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`diag`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let value = select(0.0, getX(coords[0]), coords[0] == coords[1]);
          setOutputAtIndex(index, value);
        }
      }
    `}};function Ml(e){let{inputs:t,backend:n}=e,{x:r}=t,i=[...r.shape,...r.shape],a=M(r.shape),o=J({inputs:{x:r},backend:n,attrs:{shape:[a]}}),s=new jl(a),c=n.runWebGPUProgram(s,[o],o.dtype),l=J({inputs:{x:c},backend:n,attrs:{shape:i}});return n.disposeData(o.dataId),n.disposeData(c.dataId),l}var Nl={kernelName:tr,backendName:`webgpu`,kernelFunc:Ml},Pl=class{constructor(e){this.variableNames=[`x`,`w`],this.uniforms=`filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`dilation2d`}getUserCode(){return`
       ${z(`index`)} {
         if (index < uniforms.size) {
           let neg_infinity = -3.4e38;
           let coords = getOutputCoords();
           let batch = coords.x;
           let d1 = coords.w;
           let outTopLeftCorner = coords.yz * uniforms.strides - uniforms.pads;
           let hBeg = outTopLeftCorner.x;
           let wBeg = outTopLeftCorner.y;

           var curVal = neg_infinity;
           for (var h = 0; h < uniforms.filterDims[0]; h = h + 1) {
             let hIn = hBeg + h * uniforms.dilations[0];

             if (hIn >= 0 && hIn < uniforms.xShape[1]) {
               for (var w = 0; w < uniforms.filterDims[1]; w = w + 1) {
                 let wIn = wBeg + w * uniforms.dilations[1];

                 if (wIn >= 0 && wIn < uniforms.xShape[2]) {
                   let val = getX(batch, hIn, wIn, d1) + getW(h, w, d1);
                   if (val > curVal) {
                     curVal = val;
                   }
                 }
               }
             }
           }

           setOutputAtIndex(index, curVal);
         }
       }
     `}};function Fl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a}=t,{strides:o,pad:s,dilations:c}=r,l=Nt(i.shape,a.shape,o,s,`NHWC`,c),u=[l.padInfo.top,l.padInfo.left],d=[{type:`int32`,data:[l.filterHeight,l.filterWidth]},{type:`int32`,data:[...u]},{type:`int32`,data:[l.strideHeight,l.strideWidth]},{type:`int32`,data:[l.dilationHeight,l.dilationWidth]}],f=new Pl(l);return n.runWebGPUProgram(f,[i,a],i.dtype,d)}var Il={kernelName:Mn,backendName:`webgpu`,kernelFunc:Fl},Ll=class{constructor(e,t){if(this.variableNames=[`x`,`w`,`dy`],this.uniforms=`filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,`,this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=e.inShape,this.dispatchLayout=V(e.outShape),this.dispatch=B(this.dispatchLayout,e.outShape,this.workgroupSize),t!==`float32`&&t!==`int32`)throw Error(`Dilation2DBackpropInput only supports float32 and int32
          types, does not support ${t} type.`);this.type=t,this.shaderKey=`dilation2DBackpropInput`}getUserCode(){return`
       ${z(`index`)} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var xRMax = 0;
           var xCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     xRMax = xR;
                     xCMax = xC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.xShape[3] *
               (xCMax + uniforms.xShape[2] * (xRMax + uniforms.xShape[1] * b));
           let value = getDy(b, r, c, d);
           ${F(`&result[flatIndexIn]`,`value`,this.type)}
         }
       }
     `}},Rl=class{constructor(e,t,n){if(this.variableNames=[`x`,`w`,`dy`],this.uniforms=`filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,`,this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=e.filterShape,this.dispatchLayout=V(e.outShape),this.dispatch=B(this.dispatchLayout,e.outShape,this.workgroupSize),n!==`float32`&&n!==`int32`)throw Error(`Dilation2DBackpropFilter only supports float32 and int32
          types, does not support ${n} type.`);this.type=n,this.shaderKey=`dilation2DBackpropFilter`}getUserCode(){return`
       ${z(`index`)} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var wRMax = 0;
           var wCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     wRMax = wR;
                     wCMax = wC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.wShape[2] * (wCMax + wRMax * uniforms.wShape[1]);
           let value = getDy(b, r, c, d);
           ${F(`&result[flatIndexIn]`,`value`,this.type)}
         }
       }
     `}};function zl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r,u=Nt(i.shape,a.shape,s,c,`NHWC`,l),d=a.dtype,f=new Rl(u,a.shape,d),p=[{type:`int32`,data:[u.filterHeight,u.filterWidth]},{type:`int32`,data:[u.padInfo.top,u.padInfo.left]},{type:`int32`,data:[u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.dilationHeight,u.dilationWidth]},{type:`int32`,data:[M(u.outShape)]}],m=q({backend:n,attrs:{shape:a.shape,value:0,dtype:d}});return n.runWebGPUProgram(f,[i,a,o],d,p,m)}var Bl={kernelName:Kn,backendName:`webgpu`,kernelFunc:zl};function Vl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,dy:o}=t,{strides:s,pad:c,dilations:l}=r,u=Nt(i.shape,a.shape,s,c,`NHWC`,l),d=i.dtype,f=new Ll(u,d),p=[{type:`int32`,data:[u.filterHeight,u.filterWidth]},{type:`int32`,data:[u.padInfo.top,u.padInfo.left]},{type:`int32`,data:[u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.dilationHeight,u.dilationWidth]},{type:`int32`,data:[M(u.outShape)]}],m=q({backend:n,attrs:{shape:u.inShape,value:0,dtype:d}});return n.runWebGPUProgram(f,[i,a,o],d,p,m)}var Hl={kernelName:hn,backendName:`webgpu`,kernelFunc:Vl},Ul=class{constructor(e,t,n){this.variableNames=[`Image`],this.uniforms=`alpha: f32,`,this.workgroupSize=[64,1,1],this.pixelsOpType=Pr.DRAW,this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.type=t,this.textureFormat=n,this.shaderKey=`draw_${t}_${n}`}getUserCode(){let e,t=this.type===`float32`?`value`:`value / 255.0`;return e=`
      if (uniforms.numChannels == 1) {
        rgba[0] = ${t};
        rgba[1] = ${t};
        rgba[2] = ${t};
      } else {
        rgba[d] = ${t};
      }`,`
       @group(0) @binding(0) var outImage : texture_storage_2d<${this.textureFormat}, write>;
       ${z(`index`)} {
         if (index < uniforms.size) {
           var rgba = vec4<f32>(0.0, 0.0, 0.0, uniforms.alpha);
           for (var d = 0; d < uniforms.numChannels; d = d + 1) {
             let value = f32(inBuf[index * uniforms.numChannels + d]);
             ${e}
           }
           rgba.x = rgba.x * rgba.w;
           rgba.y = rgba.y * rgba.w;
           rgba.z = rgba.z * rgba.w;
           let coords = getCoordsFromIndex(index);
           textureStore(outImage, vec2<i32>(coords.yx), rgba);
         }
       }
      `}};function Wl(e){let{inputs:t,backend:n,attrs:r}=e,{image:i}=t,{canvas:a,options:o}=r,[s,c]=i.shape.slice(0,2),{imageOptions:l}=o||{},u=l?.alpha||1,d=n.device.features.has(`bgra8unorm-storage`)?`bgra8unorm`:`rgba8unorm`,f=[s,c],p=new Ul(f,i.dtype,d);a.width=c,a.height=s;let m=`webgpu`,h=a.getContext(m),g;h||=(g=new OffscreenCanvas(c,s),g.getContext(m));let _=i.shape.length===3?i.shape[2]:1;h.configure({device:n.device,format:d,usage:GPUTextureUsage.STORAGE_BINDING,alphaMode:`premultiplied`});let v=`int32`,y=n.makeTensorInfo(f,v),b=n.tensorMap.get(y.dataId);b.resource=h.getCurrentTexture(),b.external=!0;let x=[{type:`uint32`,data:[_]},{type:`float32`,data:[u]}];if(n.runWebGPUProgram(p,[i],v,x,y),g){let e=a.getContext(`2d`);if(!e)throw Error(`Please make sure this canvas has only been used for 2d or webgpu context!`);e.drawImage(g,0,0)}return n.disposeData(y.dataId),i}var Gl={kernelName:Lt,backendName:`webgpu`,kernelFunc:Wl},Kl=Z({opType:U.MUL,cpuKernelImpl:Oo,supportsComplex:!0}),ql={kernelName:Zn,backendName:`webgpu`,kernelFunc:Kl};function Jl(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return $(i,a,o,`sum`,n)}var Yl={kernelName:`Sum`,backendName:`webgpu`,kernelFunc:Jl};function Xl(e){let{inputs:t,backend:n,attrs:r}=e,{equation:i}=r,o=t,{allDims:s,summedDims:c,idDims:u}=w(i,o.length);O(s.length,u,o);let{path:d,steps:f}=Fe(c,u),p=f.length,m=null,h=s.length,g=[];for(let e=0;e<p;++e){for(let t of f[e]){let{permutationIndices:e,expandDims:r}=Cn(h,u[t]),i;l(e)?i=o[t]:(i=Q({inputs:{x:o[t]},backend:n,attrs:{perm:e}}),g.push(i));let s=i.shape.slice();for(let e=0;e<r.length;++e)s.splice(r[e],0,1);a(i.shape,s)||(i=J({inputs:{x:i},backend:n,attrs:{shape:s}}),g.push(i)),m===null?m=i:(m=Kl({inputs:{a:i,b:m},backend:n}),g.push(m))}e<p-1&&(d[e]>=0&&(m=Jl({inputs:{x:m},backend:n,attrs:{axis:d[e]-(s.length-h),keepDims:!1}}),g.push(m)),h--)}for(let e of g)e!==m&&n.disposeData(e.dataId);return m}var Zl={kernelName:Fn,backendName:`webgpu`,kernelFunc:Xl},Ql={kernelName:`Elu`,backendName:`webgpu`,kernelFunc:X({opType:W.ELU})},$l={kernelName:Wt,backendName:`webgpu`,kernelFunc:e=>{let{inputs:t,backend:n}=e,{dy:r,y:i}=t,a=new ro(U.ELU_DER,r.shape,i.shape);return n.runWebGPUProgram(a,[r,i],r.dtype)}},eu=Z({opType:U.EQUAL,dtype:`bool`,cpuKernelImpl:po}),tu={kernelName:En,backendName:`webgpu`,kernelFunc:eu},nu={kernelName:`Erf`,backendName:`webgpu`,kernelFunc:X({opType:W.ERF})},ru={kernelName:`Exp`,backendName:`webgpu`,kernelFunc:X({opType:W.EXP,cpuKernelImpl:mo,dtype:`float32`})};function iu(e){let{inputs:t,attrs:n,backend:r}=e,{dim:i}=n,{input:a}=t,o=a.shape.length,s=a.shape.slice(),c=i;return i<0&&(k(-(o+1)<=i,()=>`Axis must be in the interval [${-(o+1)}, ${o}]`),c=o+i+1),s.splice(c,0,1),J({inputs:{x:a},backend:r,attrs:{shape:s}})}var au={kernelName:nn,backendName:`webgpu`,kernelFunc:iu},ou=X({opType:W.EXPM1,cpuKernelImpl:ho}),su={kernelName:zn,backendName:`webgpu`,kernelFunc:ou},cu=class{constructor(e,t){this.variableNames=[`real`,`imag`],this.outputShape=[],this.uniforms=`exponentMultiplier : f32, denominator: f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.component=e,this.shaderKey=`fft_${e}`}getUserCode(){return`
    fn unaryOpComplex(real: f32, expR: f32, imag: f32, expI: f32) -> f32 {
      ${this.component===`real`?`return real * expR - imag * expI;`:`return real * expI + imag * expR;`}
    }

    fn mulMatDFT(batch: i32, index: i32) -> f32 {
      let indexRatio = f32(index) / f32(uniforms.realShape[1]);
      let exponentMultiplierTimesIndexRatio =
          uniforms.exponentMultiplier * indexRatio;

      var result = 0.0;

      for (var i = 0; i < uniforms.realShape[1]; i = i + 1) {
        // x = (-2|2 * PI / N) * index * i;
        let x = exponentMultiplierTimesIndexRatio * f32(i);
        let expR = cos(x);
        let expI = sin(x);
        let real = getReal(batch, i);
        let imag = getImag(batch, i);

        result = result +
            unaryOpComplex(real, expR, imag, expI) / uniforms.denominator;
      }

      return result;
    }

    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        setOutputAtIndex(index, mulMatDFT(coords[0], coords[1]));
      }
    }
  `}};function lu(e,t,n){let r=n.tensorMap.get(e.dataId),i=M(e.shape),a=e.shape[e.shape.length-1],o=i/a,s=[],c=J({inputs:{x:e},backend:n,attrs:{shape:[o,a]}});s.push(c);let l=c.shape,u=new cu(`real`,l),d=new cu(`imag`,l),f=[{dataId:r.complexTensorInfos.real.dataId,dtype:r.complexTensorInfos.real.dtype,shape:l},{dataId:r.complexTensorInfos.imag.dataId,dtype:r.complexTensorInfos.imag.dtype,shape:l}],p=t?2*Math.PI:-2*Math.PI,m=t?l[1]:1,h=[{type:`float32`,data:[p]},{type:`float32`,data:[m]}],g=n.runWebGPUProgram(u,f,`float32`,h);s.push(g);let _=n.runWebGPUProgram(d,f,`float32`,h);s.push(_);let v=ao({inputs:{real:g,imag:_},backend:n});s.push(v);let y=J({inputs:{x:v},backend:n,attrs:{shape:e.shape}});return s.forEach(e=>n.disposeData(e.dataId)),y}function uu(e){let{inputs:t,backend:n}=e,{input:r}=t;return lu(r,!1,n)}var du={kernelName:`FFT`,backendName:`webgpu`,kernelFunc:uu},fu=class{constructor(e){this.outputShape=[],this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`flipLeftRight`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let coordX = uniforms.xShape[2] - coords[2] - 1;
          let outputValue = getX(coords[0], coords[1], coordX, coords[3]);
          setOutputAtIndex(index, outputValue);
        }
      }
    `}},pu={kernelName:sn,backendName:`webgpu`,kernelFunc:({inputs:e,backend:t})=>{let{image:n}=e,r=t,i=new fu(n.shape);return r.runWebGPUProgram(i,[n],n.dtype)}},mu=X({opType:W.FLOOR,cpuKernelImpl:go}),hu={kernelName:Mt,backendName:`webgpu`,kernelFunc:mu},gu=Z({opType:U.FLOOR_DIV,cpuKernelImpl:_o,dtype:`int32`}),_u={kernelName:cr,backendName:`webgpu`,kernelFunc:gu},vu=class{constructor(e,t,n=!1){this.pixelsOpType=Pr.FROM_PIXELS,this.outputShape=[0],this.variableNames=[],this.workgroupSize=[256,1,1],this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[t,1,1]),this.importVideo=n,this.shaderKey=`fromPixels_${this.importVideo}`}getUserCode(){let e=this.importVideo?`textureLoad(src, vec2<i32>(coords.yx));`:`textureLoad(src, vec2<i32>(coords.yx), 0)`;return`
      @binding(1) @group(0) var src: ${this.importVideo?`texture_external`:`texture_2d<f32>`};
      ${z(`index`)} {
        let flatIndex = index * uniforms.numChannels;
        if (flatIndex < uniforms.size) {
          let coords = getCoordsFromIndex(flatIndex);
          let values = ${e};
          for (var i = 0; i < uniforms.numChannels; i = i + 1) {
            result[flatIndex + i] = i32(floor(255.0 * values[i]));
          }
        }
      }
  `}},yu={kernelName:yr,backendName:`webgpu`,kernelFunc:Su},bu,xu=j().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);function Su(e){let{inputs:t,backend:n,attrs:r}=e,{pixels:i}=t,{numChannels:a}=r;if(i==null)throw Error(`pixels passed to tf.browser.fromPixels() can not be null`);let o=typeof HTMLVideoElement<`u`&&i instanceof HTMLVideoElement,s=typeof HTMLImageElement<`u`&&i instanceof HTMLImageElement,c=typeof HTMLCanvasElement<`u`&&i instanceof HTMLCanvasElement||typeof OffscreenCanvas<`u`&&i instanceof OffscreenCanvas,l=typeof ImageBitmap<`u`&&i instanceof ImageBitmap,[u,d]=o?[i.videoWidth,i.videoHeight]:[i.width,i.height],f=[d,u,a],p=j().getBool(`WEBGPU_IMPORT_EXTERNAL_TEXTURE`)&&o,m=o||s;if(l||c||m){let e;if(p)e=n.device.importExternalTexture({source:i});else{if(m){let e=j().getBool(`CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU`);(bu==null||e!==xu)&&(xu=e,bu=document.createElement(`canvas`).getContext(`2d`,{willReadFrequently:xu})),bu.canvas.width=u,bu.canvas.height=d,bu.drawImage(i,0,0,u,d),i=bu.canvas}let t=GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,r=n.textureManager.acquireTexture(f[1],f[0],`rgba8unorm`,t);n.queue.copyExternalImageToTexture({source:i},{texture:r},[f[1],f[0]]),e=r}let t=M(f),r=We(f),o=new vu(f,a,p),s=[{type:`uint32`,data:[t]},{type:`uint32`,data:[a]},{type:`uint32`,data:[...r]}],c=n.makeTensorInfo([d,u],`int32`),l=n.tensorMap.get(c.dataId);l.resource=e;let h=n.runWebGPUProgram(o,[c],`int32`,s);return n.disposeData(c.dataId),h}let h=i.data,g=h;if(a!=null&&a!==4){g=new Uint8Array(i.width*i.height*a);let e=h.length,t=0;for(let n=0;n<e;n++)n%4<a&&(g[t++]=h[n])}let _=n.makeTensorInfo(f,`int32`,new Int32Array(g));return n.uploadToGPU(_.dataId),_}var Cu=class{constructor(e,t,n,r,i){this.uniforms=`varianceEpsilon : f32,`,this.workgroupSize=[128,1,1],this.size=!0,this.variableNames=[`x`,`mean`,`variance`],N(e,t),N(e,n),this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),r!=null&&(N(e,r),this.variableNames.push(`offset`)),i!=null&&(N(e,i),this.variableNames.push(`scale`)),this.offsetShape=r,this.scaleShape=i,this.shaderKey=`batchNorm`}getUserCode(){let e=`0.0`;this.offsetShape!=null&&(e=`getOffsetByOutputIndex(index)`);let t=`1.0`;return this.scaleShape!=null&&(t=`getScaleByOutputIndex(index)`),`
      ${z(`index`)} {
        if (index < uniforms.size)
        {
          let xValue = getXByOutputIndex(index);
          let meanValue = getMeanByOutputIndex(index);
          let varianValue = getVarianceByOutputIndex(index);
          let offsetValue = ${e};
          let scaleValue = ${t};
          let inv = scaleValue * inverseSqrt(varianValue + f32(uniforms.varianceEpsilon));
          setOutputAtIndex(index,dot(vec3<f32>(xValue, -meanValue, offsetValue), vec3<f32>(inv, inv, 1.0)));
        }
      }
  `}},wu={kernelName:Bt,backendName:`webgpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r,scale:i,offset:a,mean:o,variance:s}=e,{varianceEpsilon:c}=t,l=n,u=[r,o,s],d=null;a!=null&&(d=a.shape,u.push(a));let f=null;i!=null&&(f=i.shape,u.push(i));let p=new Cu(r.shape,o.shape,s.shape,d,f),m=[{type:`float32`,data:[c]}];return l.runWebGPUProgram(p,u,r.dtype,m)}};function Tu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:f,activation:p,leakyreluAlpha:m}=r,h=Vt(u);return Fc({x:i,filter:a,convInfo:A(i.shape,a.shape,c,d,l,f,!1,h),backend:n,bias:o,preluActivationWeights:s,leakyreluAlpha:m,activation:p})}var Eu={kernelName:mr,backendName:`webgpu`,kernelFunc:Tu};function Du(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,filter:a,bias:o,preluActivationWeights:s}=t,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:f,leakyreluAlpha:p}=r,m=u;m??=[1,1],k(hr(c,m),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${m}'`);let h=A(i.shape,a.shape,c,m,l,d,!0),g=[i,a],_=o!=null,v=s!=null;_&&g.push(o),v&&g.push(s);let y=[{type:`int32`,data:[h.padInfo.top,h.padInfo.left]},{type:`int32`,data:[h.inHeight,h.inWidth]}],b;return h.outHeight>4&&h.outWidth>4&&h.strideWidth<=2&&h.inChannels===h.outChannels&&h.dilationHeight===1&&h.dilationWidth===1&&h.inChannels%4==0?(b=new xl(h,_,f,v),y.push({type:`int32`,data:[b.virtualWidth]})):(b=new Sl(h,_,f,v),y.push({type:`int32`,data:[h.filterHeight]},{type:`int32`,data:[h.filterWidth]},{type:`int32`,data:[h.strideHeight,h.strideWidth]},{type:`int32`,data:[h.dilationHeight,h.dilationWidth]})),f===`leakyrelu`&&(y.push({type:`float32`,data:[p]}),b.uniforms+=` alpha : f32,`),n.runWebGPUProgram(b,g,`float32`,y)}var Ou={kernelName:Ye,backendName:`webgpu`,kernelFunc:Du},ku=class{constructor(e,t){this.variableNames=[`A`,`indices`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`gathernd_${e}`,this.sliceDim=e,this.uniforms=`sliceDim : i32, strides : ${L(e)},`}getUserCode(){let e;return e=this.sliceDim>1?`uniforms.strides[j]`:`uniforms.strides`,`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          var flattenIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexTemp = i32(round(getIndices(coords[0], j)));
            let strideNum = ${e};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }

          setOutputAtIndex(index, getA(flattenIndex, coords[1]));
        }
      }
      `}};function Au(e){let{inputs:t,backend:n}=e,{params:r,indices:i}=t,a=i.shape,o=a[a.length-1],s=M(r.shape),[c,l,u,d]=ar(r,i),f=J({inputs:{x:i},backend:n,attrs:{shape:[l,o]}}),p=J({inputs:{x:r},backend:n,attrs:{shape:[M(r.shape)/u,u]}});if(n.shouldExecuteOnCPU([r,i])||r.dtype===`string`){let e=vo(n.readSync(i.dataId),n.bufferSync(r),r.dtype,l,o,u,d,r.shape,s);return n.makeTensorInfo(c,r.dtype,e.values)}let m=new ku(o,[l,u]),h=[{type:`int32`,data:[o]},{type:`int32`,data:d}],g=n.runWebGPUProgram(m,[p,f],p.dtype,h),_=J({inputs:{x:g},backend:n,attrs:{shape:c}});return n.disposeData(f.dataId),n.disposeData(p.dataId),n.disposeData(g.dataId),_}var ju={kernelName:y,backendName:`webgpu`,kernelFunc:Au},Mu=class{constructor(e,t){this.variableNames=[`A`,`indices`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.slice(),this.aShape=e,this.outputShape=t,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`gather`}getUserCode(){let e=Nu(this.aShape);return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let indexZ = i32(getIndices(resRC.x, resRC.z));
          let inBounds = select(0.0, 1.0, indexZ >= 0 && indexZ < uniforms.aShape[2]);
          setOutputAtIndex(index, inBounds * getA(${e}));
        }
      }
    `}};function Nu(e){let t=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],n=[];for(let r=0;r<e.length;r++)r===2?n.push(`indexZ`):n.push(`${t[r]}`);return n.join()}function Pu(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,indices:a}=t,{axis:o,batchDims:s}=r,c=An(o,i.shape)[0],l=Xe(i,a,c,s),u=M(a.shape),d=[],f=J({inputs:{x:i},backend:n,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),p=J({inputs:{x:a},backend:n,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(f),d.push(p);let m=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(n.shouldExecuteOnCPU([i,a])){let e=n.tensorMap.get(p.dataId).values,t=Oe(p.shape,p.dtype,e),r=n.tensorMap.get(f.dataId).values,i=yo(Oe(f.shape,f.dtype,r),t,m);return d.forEach(e=>n.disposeData(e.dataId)),n.makeTensorInfo(l.outputShape,i.dtype,i.values)}let h=new Mu(f.shape,m),g=n.runWebGPUProgram(h,[f,p],f.dtype);d.push(g);let _=J({inputs:{x:g},backend:n,attrs:{shape:l.outputShape}});return d.forEach(e=>n.disposeData(e.dataId)),_}var Fu={kernelName:dr,backendName:`webgpu`,kernelFunc:Pu},Iu=Z({opType:U.GREATER,cpuKernelImpl:xo,dtype:`bool`}),Lu={kernelName:et,backendName:`webgpu`,kernelFunc:Iu},Ru=Z({opType:U.GREATER_EQUAL,dtype:`bool`,cpuKernelImpl:bo}),zu={kernelName:ee,backendName:`webgpu`,kernelFunc:Ru};function Bu(e){let{inputs:t,backend:n}=e,{input:r}=t;return lu(r,!0,n)}var Vu={kernelName:S,backendName:`webgpu`,kernelFunc:Bu},Hu=X({opType:W.IS_FINITE,dtype:`bool`}),Uu={kernelName:s,backendName:`webgpu`,kernelFunc:Hu},Wu=X({opType:W.IS_INF,dtype:`bool`}),Gu={kernelName:vn,backendName:`webgpu`,kernelFunc:Wu},Ku=X({opType:W.IS_NAN,dtype:`bool`}),qu={kernelName:De,backendName:`webgpu`,kernelFunc:Ku};function Ju(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{alpha:a}=r,o=[{type:`float32`,data:[a]}],s=new so(i.shape,W.LEAKYRELU,`alpha : f32,`);return n.runWebGPUProgram(s,[i],`float32`,o)}var Yu={kernelName:re,backendName:`webgpu`,kernelFunc:Ju},Xu=Z({opType:U.LESS,dtype:`bool`,cpuKernelImpl:Co}),Zu={kernelName:pe,backendName:`webgpu`,kernelFunc:Xu},Qu=Z({opType:U.LESS_EQUAL,dtype:`bool`,cpuKernelImpl:So}),$u={kernelName:we,backendName:`webgpu`,kernelFunc:Qu},ed=class{constructor(e){this.variableNames=[],this.outputShape=[],this.uniforms=`start : f32, step : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`linSpace`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          setOutputAtIndex(index, uniforms.start + f32(index) * uniforms.step);
        }
      }
    `}};function td(e){let{backend:t,attrs:n}=e,{start:r,stop:i,num:a}=n,o=(i-r)/(a-1),s=new ed(a),c=[{type:`float32`,data:[r]},{type:`float32`,data:[o]}];return t.runWebGPUProgram(s,[],`float32`,c)}var nd={kernelName:Ge,backendName:`webgpu`,kernelFunc:td},rd={kernelName:`Log`,backendName:`webgpu`,kernelFunc:X({opType:W.LOG,cpuKernelImpl:wo})},id=X({opType:W.LOG1P}),ad={kernelName:p,backendName:`webgpu`,kernelFunc:id},od=Z({opType:U.LOGICAL_AND,dtype:`bool`}),sd={kernelName:le,backendName:`webgpu`,kernelFunc:od},cd=X({opType:W.LOGICAL_NOT}),ld={kernelName:it,backendName:`webgpu`,kernelFunc:cd},ud=Z({opType:U.LOGICAL_OR}),dd={kernelName:pt,backendName:`webgpu`,kernelFunc:ud},fd=`
  var powValue = 0.0;
  let basis = uniforms.bias + uniforms.alpha * sum;
  if (uniforms.beta == 0.5) {
    powValue = inverseSqrt(basis);
  } else if (uniforms.beta == 1.0) {
    powValue = 1.0 / basis;
  } else {
    powValue = exp(log(basis) * (-uniforms.beta));
  }
`,pd=class{constructor(e){this.outputShape=[],this.variableNames=[`x`],this.uniforms=`radius : i32, bias : f32, alpha : f32, beta : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`lrn`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];
        let d = coords[3];

        let x = getX(b, r, c, d);
        var sum = 0.0;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let idx = d + i;
          if (idx >= 0 && idx < uniforms.xShape[3]) {
            let z = getX(b, r, c, idx);
            sum = sum + z * z;
          }
        }
        ${fd}

        setOutputAtIndex(index, x * powValue);
      }
    }
  `}},md=class{constructor(e,t){this.outputShape=[],this.variableNames=[`x`],this.uniforms=`radius : i32, bias : f32, alpha : f32, beta : f32,`,this.workgroupSize=[256,1,1],this.maxAllowRadius=16,k(t<=this.maxAllowRadius,()=>`Radius must be less than or equal to ${this.maxAllowRadius}, current radius is ${t}`),this.outputShape=e,this.elementsPerWorkgroup=this.workgroupSize[0]-2*this.maxAllowRadius,this.dispatchLayout={x:[3],y:[2],z:[0,1]},this.dispatch=B(this.dispatchLayout,this.outputShape,[this.elementsPerWorkgroup,this.workgroupSize[1],this.workgroupSize[2]]),this.shaderKey=`lrn_shared`}getUserCode(){return`
    var <workgroup>lrnSub: array<f32, ${this.workgroupSize[0]}>;
    const elementsPerWorkgroup = ${this.elementsPerWorkgroup};
    const maxAllowRadius = ${this.maxAllowRadius};

    ${z()} {
      let localDepth = i32(localId.x);
      let workgroupDepth = i32(workgroupId.x) * elementsPerWorkgroup;
      let xDepth = workgroupDepth + localDepth - maxAllowRadius;
      let b = i32(globalId.z) / uniforms.xShape[1];
      let r = i32(globalId.z) - b * uniforms.xShape[1];
      let c = i32(globalId.y);
      let d = workgroupDepth + localDepth;

      var x = 0.0;
      if (xDepth >= 0 && xDepth < uniforms.xShape[3]) {
        x = getX(b, r, c, xDepth);
      }
      lrnSub[localDepth] = x;
      workgroupBarrier();

      if (localDepth < elementsPerWorkgroup && d < uniforms.outShape[3]) {
        var sum = 0.0;
        let index = localDepth + maxAllowRadius;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let z = lrnSub[index + i];
          sum = sum + z * z;
        }
        ${fd}

        setOutputAtCoords(b, r, c, d, lrnSub[index] * powValue);
      }
    } `}};function hd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{depthRadius:a,bias:o,alpha:s,beta:c}=r,l;l=a>16?new pd(i.shape):new md(i.shape,a);let u=[{type:`int32`,data:[a]},{type:`float32`,data:[o]},{type:`float32`,data:[s]},{type:`float32`,data:[c]}];return n.runWebGPUProgram(l,[i],i.dtype,u)}var gd={kernelName:`LRN`,backendName:`webgpu`,kernelFunc:hd},_d=class{constructor(e){this.outputShape=[],this.variableNames=[`inputImage`,`outputImage`,`dy`],this.uniforms=`depthRadius : i32, bias : f32, alpha : f32, beta : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`lrn_grad`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];

        let MIN_DEPTH_BEGIN = 0;
        let MAX_DEPTH_END = uniforms.outShape[3];
        var result = 0.0;
        for (var d = MIN_DEPTH_BEGIN; d < MAX_DEPTH_END; d++) {
          let depthBegin = max(MIN_DEPTH_BEGIN, d - uniforms.depthRadius);
          let depthEnd = min(MAX_DEPTH_END, d + uniforms.depthRadius + 1);

          var norm = 0.0;
          for (var k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; k++) {
            if (k < depthBegin) {
              continue;
            } else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            } else {
              break;
            }
          }

          norm = uniforms.alpha * norm + uniforms.bias;

          for (var k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; k++) {
            if (k < depthBegin) {
              continue;
            } else if (k >= depthBegin && k < depthEnd) {
              var dyi = -2.0 * uniforms.alpha * uniforms.beta
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d) / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * uniforms.beta);
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            } else {
              break;
            }
          }
        }

        setOutputAtIndex(index, result);
      }
    }
  `}};function vd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,y:a,dy:o}=t,{depthRadius:s,bias:c,alpha:l,beta:u}=r,d=new _d(i.shape),f=[{type:`int32`,data:[s]},{type:`float32`,data:[c]},{type:`float32`,data:[l]},{type:`float32`,data:[u]}];return n.runWebGPUProgram(d,[i,a,o],i.dtype,f)}var yd={kernelName:Le,backendName:`webgpu`,kernelFunc:vd},bd=Z({opType:U.MAX,cpuKernelImpl:Eo}),xd={kernelName:en,backendName:`webgpu`,kernelFunc:bd};function Sd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dimRoundingMode:c}=r;return js(i,lr(i.shape,a,o,1,s,c),`max`,n)}var Cd={kernelName:wt,backendName:`webgpu`,kernelFunc:Sd};function wd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{filterSize:a,strides:o,pad:s,dataFormat:c,dimRoundingMode:l}=r,u=br(i.shape,a,o,[1,1,1],s,l,c),d=new Es(u,`max`),f=[{type:`int32`,data:[u.strideDepth,u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.padInfo.front,u.padInfo.top,u.padInfo.left]},{type:`int32`,data:[u.inDepth,u.inHeight,u.inWidth]},{type:`int32`,data:[u.effectiveFilterDepth,u.effectiveFilterHeight,u.effectiveFilterWidth]}];return n.runWebGPUProgram(d,[i],i.dtype,f)}var Td={kernelName:vt,backendName:`webgpu`,kernelFunc:wd},Ed=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.uniforms=`strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`maxPool2DBackprop`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d = coords[3];

        let dyRCCorner = vec2<i32>(coords.yz) - uniforms.pads;
        let dyRCorner = dyRCCorner.x;
        let dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] - 1;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR += uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC += uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);
            let maxPosValue = lastIndex - i32(getMaxPos(batch, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            let curPosValue = wR * uniforms.filterDims[1] + wC;
            let mask = select(0.0, 1.0, maxPosValue == curPosValue);
            dotProd += dyValue * mask;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `}},Dd=class{constructor(e){this.variableNames=[`dy`,`maxPos`],this.uniforms=`strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
      outDepth : i32, outHeight : i32, outWidth : i32`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e.inShape,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`maxPool3DBackprop`}getUserCode(){return`
      ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let ch = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyDCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] * uniforms.filterDims[2] - 1;

        for (var wD = 0; wD < uniforms.filterDims[0]; wD++) {
          let dyD = f32(dyDCorner + wD) / f32(uniforms.strides[0]);

          if (dyD < 0.0 || dyD >= f32(uniforms.outDepth) || fract(dyD) > 0.0) {
            continue;
          }
          let idyD = i32(dyD);

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let dyValue = getDy(batch, idyD, idyR, idyC, ch);
              let maxPosValue = lastIndex - i32(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              let curPosValue = wD * uniforms.filterDims[1] * uniforms.filterDims[2] + wR * uniforms.filterDims[2] + wC;
              let mask = select(0.0, 1.0, maxPosValue == curPosValue);
              dotProd += dyValue * mask;
            }
          }
        }

        setOutputAtIndex(index, dotProd);
      }
    }
    `}};function Od(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a}=t,o=a,{filterSize:s,strides:c,pad:l,dimRoundingMode:u}=r,d=br(o.shape,s,c,[1,1,1],l,u),f=new Es(d,`max`,!0),p=[{type:`int32`,data:[d.strideDepth,d.strideHeight,d.strideWidth]},{type:`int32`,data:[d.padInfo.front,d.padInfo.top,d.padInfo.left]},{type:`int32`,data:[d.inDepth,d.inHeight,d.inWidth]},{type:`int32`,data:[d.effectiveFilterDepth,d.effectiveFilterHeight,d.effectiveFilterWidth]}],m=n.runWebGPUProgram(f,[o],`int32`,p),h=new Dd(d);p=[{type:`int32`,data:[d.strideDepth,d.strideHeight,d.strideWidth]},{type:`int32`,data:[d.effectiveFilterDepth-1-d.padInfo.front,d.effectiveFilterHeight-1-d.padInfo.top,d.effectiveFilterWidth-1-d.padInfo.left]},{type:`int32`,data:[d.effectiveFilterDepth,d.effectiveFilterHeight,d.effectiveFilterWidth]},{type:`int32`,data:[d.outDepth]},{type:`int32`,data:[d.outHeight]},{type:`int32`,data:[d.outWidth]}];let g=n.runWebGPUProgram(h,[i,m],o.dtype,p);return n.disposeData(m.dataId),g}var kd={kernelName:Ot,backendName:`webgpu`,kernelFunc:Od};function Ad(e){let{inputs:t,backend:n,attrs:r}=e,{dy:i,input:a,output:o}=t,s=a;si([a,o],`maxPoolGrad`);let{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=r,f=lr(s.shape,c,l,1,u,d),p=new Ts(f,`max`,!0),m=[{type:`int32`,data:[f.strideHeight,f.strideWidth]},{type:`int32`,data:[f.padInfo.top,f.padInfo.left]},{type:`int32`,data:[f.dilationHeight,f.dilationWidth]},{type:`int32`,data:[f.inHeight,f.inWidth]},{type:`int32`,data:[f.effectiveFilterHeight,f.effectiveFilterWidth]}],h=n.runWebGPUProgram(p,[s],`int32`,m),g=new Ed(f);m=[{type:`int32`,data:[f.strideHeight,f.strideWidth]},{type:`int32`,data:[f.effectiveFilterHeight-1-f.padInfo.top,f.effectiveFilterWidth-1-f.padInfo.left]},{type:`int32`,data:[f.dilationHeight,f.dilationWidth]},{type:`int32`,data:[f.effectiveFilterHeight,f.effectiveFilterWidth]},{type:`int32`,data:[f.outHeight]},{type:`int32`,data:[f.outWidth]}];let _=n.runWebGPUProgram(g,[i,h],s.dtype,m);return n.disposeData(h.dataId),_}var jd={kernelName:Ve,backendName:`webgpu`,kernelFunc:Ad};function Md(e){let{inputs:t,backend:n,attrs:r}=e,{filterSize:i,strides:a,pad:o,includeBatchInIndex:s}=r,{x:c}=t;k(c.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${c.shape.length}.`);let l=[1,1];k(hr(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);let u=lr(c.shape,i,a,l,o),d=[{type:`int32`,data:[u.strideHeight,u.strideWidth]},{type:`int32`,data:[u.padInfo.top,u.padInfo.left]},{type:`int32`,data:[u.dilationHeight,u.dilationWidth]},{type:`int32`,data:[u.inHeight,u.inWidth]},{type:`int32`,data:[u.effectiveFilterHeight,u.effectiveFilterWidth]}],f=new Ts(u,`max`,!1),p=n.runWebGPUProgram(f,[c],c.dtype,d);return f=new Ts(u,`max`,!0,!0,s),[p,n.runWebGPUProgram(f,[c],`int32`,d)]}var Nd={kernelName:r,backendName:`webgpu`,kernelFunc:Md};function Pd(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return $(i,a,o,`min`,n)}var Fd={kernelName:`Min`,backendName:`webgpu`,kernelFunc:Pd},Id=Z({opType:U.MIN,cpuKernelImpl:Do}),Ld={kernelName:Jn,backendName:`webgpu`,kernelFunc:Id},Rd=class{constructor(e,t,n){this.uniforms=``,this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]),this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.xShape=e,t.map((e,t)=>{this.uniforms+=` pad${t} : vec2<i32>,`}),this.offset=n===`reflect`?0:1,this.shaderKey=`mirrorPad_${n}`}getUserCode(){let e=this.xShape.length,t=this.xShape.map((e,t)=>`uniforms.pad${t}[0]`).join(`,`),n=this.xShape.map((t,n)=>`uniforms.pad${n}[0] + uniforms.xShape${e>1?`[${n}]`:``}`).join(`,`),r=e===1?`start`:`start[i]`,i=e===1?`end`:`end[i]`,a=e===1?`outC`:`outC[i]`,o=L(e),s=e>1?[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,e):`coords`;return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let start = ${o}(${t});
          let end = ${o}(${n});
          var outC = getCoordsFromIndex(index);
          for (var i = 0; i < ${e}; i = i + 1) {
            if (${a} < ${r}) {
              ${a} = ${r} * 2 - ${a} - ${this.offset};
            } else if(${a} >= ${i}) {
              ${a} = (${i} - 1) * 2 - ${a} + ${this.offset};
            }
          }
          let coords = outC - start;
          setOutputAtIndex(index, getX(${s}));
        }
      }
    `}},zd={kernelName:gn,backendName:`webgpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{x:r}=e,{paddings:i,mode:a}=t,o=n,s=i.map(e=>({type:`int32`,data:[e[0],e[1]]})),c=new Rd(r.shape,i,a);return o.runWebGPUProgram(c,[r],r.dtype,s)}},Bd={kernelName:`Mod`,backendName:`webgpu`,kernelFunc:Z({opType:U.MOD})},Vd=class{constructor(e,t){this.variableNames=[`probs`],this.outputShape=[],this.uniforms=`seed : f32, numOutcomes: i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e,t],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`multinomial`}getUserCode(){return`
    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    fn random (seed : f32, resultUV : vec2<f32>) -> f32 {
      let HASHSCALE1 = 443.8975;
      let p = resultUV * seed;
      var p3  = fract(vec3<f32>(p.xyx) * HASHSCALE1);
      p3 = p3 + dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${z(`index`)} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords[0];

        let resUV = vec2<f32>(f32(coords[1]) / f32(uniforms.outShape[1]),
            f32(coords[0]) / f32(uniforms.outShape[0]));
        let r = random(uniforms.seed, resUV);
        var cdf = 0.0;
        for (var i = 0; i < uniforms.numOutcomes - 1; i = i + 1) {
          cdf = cdf + getProbs(batch, i);

          if (r < cdf) {
            setOutputAtIndexI32(index, i);
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutputAtIndexI32(index, uniforms.numOutcomes - 1);
      }
    }
  `}},Hd=class{constructor(e){this.variableNames=[`logits`],this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=[this.outputShape[0],1,1],this.workgroupSize=this.outputShape[1]>=4096?[256,1,1]:[64,1,1],this.shaderKey=`softmax`}getUserCode(){return`
    var<workgroup> buf : array<f32, ${this.workgroupSize[0]}>;
    var<workgroup> rowMaxShared : f32;
    var<workgroup> rowSumShared : f32;
    const blockSize = ${this.workgroupSize[0]};
    ${z(`index`)} {
      let row = index / blockSize;
      let tid = i32(localId.x);
      let cols = uniforms.outShape[1];

      var threadMax = -3.402823e+38f;
      for (var col = tid; col < cols; col += blockSize) {
        let value = getLogits(row, col);
        threadMax = max(threadMax, value);
      }
      if (tid < cols) {
        buf[tid] = threadMax;
      }
      workgroupBarrier();

      var reduceSize = min(cols, blockSize);
      for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
        reduceSize = currSize + (reduceSize & 1);
        if (tid < currSize) {
          buf[tid] = max(buf[tid], buf[tid + reduceSize]);
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowMaxShared = buf[0];
      }
      workgroupBarrier();

      var threadSum = 0.0;
      for (var col = tid; col < cols; col += blockSize) {
        let subExp = exp(getLogits(row, col) - rowMaxShared);
        threadSum += subExp;
      }
      buf[tid] = threadSum;
      workgroupBarrier();

      for (var currSize = blockSize >> 1;  currSize > 0; currSize = currSize >> 1) {
        if (tid < currSize) {
          buf[tid] = buf[tid] + buf[tid + currSize];
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowSumShared = buf[0];
      }
      workgroupBarrier();

      for (var col = tid; col < cols; col += blockSize) {
        let value = exp(getLogits(row, col) - rowMaxShared) / rowSumShared;
        setOutputAtCoords(row, col, value);
      }
  }
    `}};function Ud(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{dim:a}=r,o=J({inputs:{x:i},backend:n,attrs:{shape:[M(i.shape)/i.shape[a],i.shape[a]]}}),s=new Hd(o.shape),c=n.runWebGPUProgram(s,[o],i.dtype),l=J({inputs:{x:c},backend:n,attrs:{shape:i.shape}});return n.disposeData(o.dataId),n.disposeData(c.dataId),l}var Wd={kernelName:ge,backendName:`webgpu`,kernelFunc:Ud};function Gd(e){let{inputs:t,backend:n,attrs:r}=e,{logits:i}=t,{numSamples:a,seed:o,normalized:s}=r,c=s?i:Ud({inputs:{logits:i},backend:n,attrs:{dim:i.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new Vd(l,a),f=[{type:`float32`,data:[o]},{type:`int32`,data:[u]}],p=n.runWebGPUProgram(d,[c],`int32`,f);return s||n.disposeData(c.dataId),p}var Kd={kernelName:In,backendName:`webgpu`,kernelFunc:Gd};function qd(e){let{inputs:t,backend:n}=e,{x:r}=t;if(n.shouldExecuteOnCPU([r])){let[e,t]=ko(n.tensorMap.get(r.dataId).values,r.shape,r.dtype);return n.makeTensorInfo(t,r.dtype,e)}let i=new so(r.shape,W.NEG);return n.runWebGPUProgram(i,[r],r.dtype)}var Jd={kernelName:`Neg`,backendName:`webgpu`,kernelFunc:qd};function Yd(e){console.warn(`tf.nonMaxSuppression() in webgpu locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c}=r,l=n.readSync(i.dataId),u=n.readSync(a.dataId),{selectedIndices:d}=v(l,u,o,s,c);return n.makeTensorInfo([d.length],`int32`,new Int32Array(d))}var Xd={kernelName:Dn,backendName:`webgpu`,kernelFunc:Yd};function Zd(e){console.warn(`tf.nonMaxSuppression() in webgpu locks the UI thread. Call tf.nonMaxSuppressionAsync() instead`);let{inputs:t,backend:n,attrs:r}=e,{boxes:i,scores:a}=t,{maxOutputSize:o,iouThreshold:s,scoreThreshold:c,softNmsSigma:l}=r,u=n.readSync(i.dataId),d=n.readSync(a.dataId),{selectedIndices:f,selectedScores:p}=$e(u,d,o,s,c,l);return[n.makeTensorInfo([f.length],`int32`,new Int32Array(f)),n.makeTensorInfo([p.length],`float32`,new Float32Array(p))]}var Qd={kernelName:Jt,backendName:`webgpu`,kernelFunc:Zd},$d=class{constructor(e,t){this.variableNames=[`x`],this.uniforms=`onValue : f32, offValue : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e,t],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`onehot`}getUserCode(){return`
      ${z(`index`)} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          setOutputAtIndex(index, mix(uniforms.offValue, uniforms.onValue,
                                      f32(i32(round(getX(coords.x))) == coords.y)));
        }
      }
    `}};function ef(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i}=t,{dtype:a,depth:o,onValue:s,offValue:c}=r,l=M(i.shape),u=new $d(l,o),d=J({inputs:{x:i},backend:n,attrs:{shape:[l]}}),f=[{type:`float32`,data:[s]},{type:`float32`,data:[c]}],p=n.runWebGPUProgram(u,[d],a,f);n.disposeData(d.dataId);let m=[...i.shape,o],h=J({inputs:{x:p},backend:n,attrs:{shape:m}});return n.disposeData(p.dataId),h}var tf={kernelName:Bn,backendName:`webgpu`,kernelFunc:ef};function nf(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`complex64`){let e=oc({inputs:{input:r},backend:n}),t=nf({inputs:{x:e},backend:n}),i=Sc({inputs:{input:r},backend:n}),a=nf({inputs:{x:i},backend:n}),o=ao({inputs:{real:t,imag:a},backend:n});return n.disposeData(e.dataId),n.disposeData(t.dataId),n.disposeData(i.dataId),n.disposeData(a.dataId),o}return q({attrs:{shape:r.shape,dtype:r.dtype,value:r.dtype===`string`?``:0},backend:n})}var rf={kernelName:Qe,backendName:`webgpu`,kernelFunc:nf};function af(e){let{inputs:t,backend:n}=e,{x:r}=t;if(r.dtype===`string`)throw Error(`onesLike is not supported under string dtype`);if(r.dtype===`complex64`){let e=oc({inputs:{input:r},backend:n}),t=af({inputs:{x:e},backend:n}),i=Sc({inputs:{input:r},backend:n}),a=nf({inputs:{x:i},backend:n}),o=ao({inputs:{real:t,imag:a},backend:n});return n.disposeData(e.dataId),n.disposeData(t.dataId),n.disposeData(i.dataId),n.disposeData(a.dataId),o}return q({attrs:{shape:r.shape,dtype:r.dtype,value:1},backend:n})}var of={kernelName:On,backendName:`webgpu`,kernelFunc:af};function sf(e){let{inputs:t,backend:n,attrs:r}=e,{axis:i}=r;if(t.length===1)return iu({inputs:{input:t[0]},backend:n,attrs:{dim:i}});let a=t[0].shape,o=t[0].dtype;t.forEach(e=>{ne(a,e.shape,`All tensors passed to stack must have matching shapes`),k(o===e.dtype,()=>`All tensors passed to stack must have matching dtypes`)});let s=[],c=Ec({inputs:t.map(e=>{let t=iu({inputs:{input:e},backend:n,attrs:{dim:i}});return s.push(t),t}),backend:n,attrs:{axis:i}});return s.forEach(e=>n.disposeData(e.dataId)),c}var cf={kernelName:fn,backendName:`webgpu`,kernelFunc:sf};function lf(e,t=!1){let n=e.length,r=L(n),i=e.map((e,t)=>`uniforms.pad${t}[0]`).join(`,`),a=e.map((e,t)=>`uniforms.pad${t}[0] + uniforms.xShape${n>1?`[${t}]`:``}`).join(`,`),o=n>1?`${r}(${i})`:`${i}`,s=n>1?`${r}(${a})`:`${a}`,c=n>1?`any(paddedCoords < start)`:`paddedCoords < start`,l=n>1?`any(paddedCoords >= end)`:`paddedCoords >= end`,u=n>1?[`coords[0]`,`coords[1]`,`coords[2]`,`coords[3]`].slice(0,n):`coords`;return`
        let start = ${o};
        let end = ${s};
        if (${c} || ${l}) {
          setOutputAtIndex(index, ${t?0:`uniforms.constantValue`});
        } else {
          let coords = paddedCoords - start;
          setOutputAtIndex(index, getX(${u}));
        }
  `}var uf=class{constructor(e,t){this.variableNames=[`x`],this.uniforms=`constantValue : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((t,n)=>t[0]+e[n]+t[1]),this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),t.map((e,t)=>{this.uniforms+=` pad${t} : vec2<i32>,`}),this.xShape=e,this.shaderKey=`pad`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let paddedCoords = getCoordsFromIndex(index);
          ${lf(this.xShape)}
        }
      }
    `}},df={kernelName:ln,backendName:`webgpu`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{paddings:o,constantValue:s}=r;if(o.every(e=>a(e,[0,0])))return Y({inputs:{x:i},backend:n});if(M(i.shape)===0)return q({backend:n,attrs:{shape:o.map((e,t)=>e[0]+i.shape[t]+e[1]),value:s,dtype:i.dtype}});let c=[{type:`float32`,data:[s]}];o.map(e=>c.push({type:`int32`,data:[e[0],e[1]]}));let l=new uf(i.shape,o);return n.runWebGPUProgram(l,[i],i.dtype,c)}},ff={kernelName:`Pow`,backendName:`webgpu`,kernelFunc:Z({opType:U.POW})};function pf(e){let{inputs:t,backend:n}=e,{x:r,alpha:i}=t,a=new ro(U.PRELU,r.shape,i.shape);return n.runWebGPUProgram(a,[r,i],`float32`)}var mf={kernelName:xr,backendName:`webgpu`,kernelFunc:pf};function hf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{axis:a,keepDims:o}=r;return $(i,a,o,`prod`,n)}var gf={kernelName:Ht,backendName:`webgpu`,kernelFunc:hf},_f={kernelName:fr,backendName:`webgpu`,kernelFunc:e=>{let{backend:t,attrs:n}=e,{start:r,stop:i,step:a,dtype:o}=n,s=Mo(r,i,a,o);return t.makeTensorInfo([s.length],o,s)}},vf=Z({opType:U.DIV}),yf={kernelName:D,backendName:`webgpu`,kernelFunc:vf},bf=X({opType:W.RECIPROCAL}),xf={kernelName:C,backendName:`webgpu`,kernelFunc:bf},Sf=X({opType:W.RELU}),Cf={kernelName:Pe,backendName:`webgpu`,kernelFunc:Sf},wf=X({opType:W.RELU6}),Tf={kernelName:Sn,backendName:`webgpu`,kernelFunc:wf},Ef=class{constructor(e,t,n){this.variableNames=[`x`],this.uniforms=`adjustHeightWidth : vec2<f32>, halfPixelCenters : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e[0],t,n,e[3]],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`resizeBilinear`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            f32(uniforms.xShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.xShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveOutSize = vec2<f32>(
            f32(uniforms.outShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.outShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC =
            (vec2<f32>(rc) + vec2<f32>(uniforms.halfPixelCenters)) *
            effectiveInputOverOutputRatioRC - vec2<f32>(uniforms.halfPixelCenters);

          // Compute the four integer indices.
          let sourceFloorRC = vec2<i32>(sourceFracIndexRC);
          let sourceCeilRC = vec2<i32>(
            min(vec2<f32>(uniforms.xShape.yz) - vec2<f32>(1.0), ceil(sourceFracIndexRC)));

          let topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          let bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          let topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          let bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          let fracRC = sourceFracIndexRC - vec2<f32>(sourceFloorRC);

          let top = topLeft + (topRight - topLeft) * fracRC.y;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          let newValue = top + (bottom - top) * fracRC.x;

          setOutputAtIndex(index, newValue);
        }
      }
    `}};function Df(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,size:o,halfPixelCenters:s}=r,[c,l]=o,u=[{type:`float32`,data:[a&&c>1?1:0,a&&l>1?1:0]},{type:`float32`,data:[s?.5:0]}],d=new Ef(i.shape,c,l);return n.runWebGPUProgram(d,[i],`float32`,u)}var Of={kernelName:yn,backendName:`webgpu`,kernelFunc:Df},kf=class{constructor(e,t){this.variableNames=[`dy`],this.uniforms=`effectiveXSize : vec2<i32>, effectiveYSize : vec2<i32>, heightScale : f32, widthScale : f32,
       invHeightScale : f32, invWidthScale : f32, winHeight : i32, winWidth : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.alignCorners=t,this.shaderKey=`resizeBilinearBackprop_${t}`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let b = coords[0];
          let d = coords[3];
          let r = coords[1];
          let c = coords[2];

          var accumulator = 0.0;

          // Compute bounds for where in dy we will look
          let startRLerp = floor(f32(r) * uniforms.invHeightScale);
          let startDyR = i32(startRLerp - f32(uniforms.winHeight / 2));

          let startCLerp = floor(f32(c) * uniforms.invWidthScale);
          let startDyC = i32(startCLerp - f32(uniforms.winWidth / 2));

          // Loop over dy
          for (var dyROffset = 0; dyROffset < uniforms.winHeight; dyROffset++) {
            let dyR = startDyR + dyROffset;

            // Guard against the window exceeding the bounds of dy
            if (dyR < 0 || dyR >= uniforms.dyShape[1]) {
              continue;
            }

            for (var dyCOffset = 0; dyCOffset < uniforms.winWidth; dyCOffset++) {
              let dyC = startDyC + dyCOffset;

              // Guard against the window exceeding the bounds of dy
              if (dyC < 0 || dyC >= uniforms.dyShape[2]) {
                continue;
              }

              let dxR = f32(dyR) * uniforms.heightScale;
              let topDxRIndex = i32(floor(dxR));
              let bottomDxRIndex = i32(min(ceil(dxR), f32(uniforms.outShape[1] - 1)));
              let dxRLerp = dxR - f32(topDxRIndex);
              let inverseDxRLerp = 1.0 - dxRLerp;

              let dxC = f32(dyC) * uniforms.widthScale;
              let leftDxCIndex = i32(floor(dxC));
              let rightDxCIndex = i32(min(ceil(dxC), f32(uniforms.outShape[2] - 1)));
              let dxCLerp = dxC - f32(leftDxCIndex);
              let inverseDxCLerp = 1.0 - dxCLerp;

              if (r == topDxRIndex && c == leftDxCIndex) {
                // topLeft
                accumulator +=
                  getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
              }

              if (r == topDxRIndex && c == rightDxCIndex) {
                // topRight
                accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
              }

              if (r == bottomDxRIndex && c == leftDxCIndex) {
                // bottomLeft
                accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
              }

              if (r == bottomDxRIndex && c == rightDxCIndex) {
                // bottomRight
                accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
              }
            }
          }
          // End loop over dy

          setOutputAtIndex(index, accumulator);
        }
      }
    `}};function Af(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,[,s,c]=i.shape,[,l,u]=a.shape,d=[o&&l>1?s-1:s,o&&u>1?c-1:c],f=[o&&l>1?l-1:l,o&&u>1?u-1:u],p=d[0]/f[0],m=d[1]/f[1],h=1/p,g=1/m,_=Math.ceil(h)*2+2,v=Math.ceil(g)*2+2,y=new kf(i.shape,o),b=[{type:`int32`,data:d},{type:`int32`,data:f},{type:`float32`,data:[p]},{type:`float32`,data:[m]},{type:`float32`,data:[h]},{type:`float32`,data:[g]},{type:`int32`,data:[_]},{type:`int32`,data:[v]}];return n.runWebGPUProgram(y,[a],a.dtype,b)}var jf={kernelName:ke,backendName:`webgpu`,kernelFunc:Af},Mf=class{constructor(e,t,n,r){this.variableNames=[`x`],this.uniforms=`adjustHeightWidth : vec2<f32>, roundBase : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e[0],t,n,e[3]],this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.halfPixelCenters=r,this.shaderKey=`resizeNearest_${r}`}getUserCode(){let e;return e=this.halfPixelCenters?`max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC, vec2<f32>(0.0))`:`vec2<f32>(rc) * effectiveInputOverOutputRatioRC`,`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            f32(uniforms.xShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.xShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveOutSize = vec2<f32>(
            f32(uniforms.outShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.outShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = ${e};

          // Compute the coordinators of nearest neighbor point.
          let inputShapeRC = vec2<f32>(f32(uniforms.xShape.y), f32(uniforms.xShape.z));
          let sourceNearestRC = vec2<i32>(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + uniforms.roundBase)));
          let newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutputAtIndex(index, newValue);
        }
      }
    `}};function Nf(e){let{inputs:t,backend:n,attrs:r}=e,{images:i}=t,{alignCorners:a,halfPixelCenters:o,size:s}=r,[c,l]=s,u=[{type:`float32`,data:[a&&c>1?1:0,a&&l>1?1:0]},{type:`float32`,data:[a?.5:0]}],d=new Mf(i.shape,c,l,o);return n.runWebGPUProgram(d,[i],i.dtype,u)}var Pf={kernelName:je,backendName:`webgpu`,kernelFunc:Nf},Ff=class{constructor(e,t){this.variableNames=[`dy`],this.uniforms=`effectiveXSize : vec2<i32>, effectiveYSize : vec2<i32>, invHeightScale : f32, invWidthScale : f32,
       winHeight : i32, winWidth : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.alignCorners=t,this.shaderKey=`resizeNearestNeigborBackprop_${t}`}getUserCode(){return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let b = coords[0];
          let d = coords[3];
          let r = coords[1];
          let c = coords[2];

          var accumulator = 0.0;

          // Compute bounds for where in dy we will look
          let startRLerp = floor(f32(r) * uniforms.invHeightScale);
          let startDyR = i32(floor(startRLerp - f32(uniforms.winHeight / 2)));

          let startCLerp = floor(f32(c) * uniforms.invWidthScale);
          let startDyC = i32(floor(startCLerp - f32(uniforms.winWidth / 2)));

          // Loop over dy
          for (var dyROffset = 0; dyROffset < uniforms.winHeight; dyROffset++) {
            let dyR = startDyR + dyROffset;

            // Guard against the window exceeding the bounds of dy
            if (dyR < 0 || dyR >= uniforms.dyShape[1]) {
              continue;
            }

            for (var dyCOffset = 0; dyCOffset < uniforms.winWidth; dyCOffset++) {
              let dyC = startDyC + dyCOffset;

              // Guard against the window exceeding the bounds of dy
              if (dyC < 0 || dyC >= uniforms.dyShape[2]) {
                continue;
              }

              let sourceFracRow = f32(uniforms.effectiveXSize[0]) *
                  (f32(dyR) / f32(uniforms.effectiveYSize[0]));

              let sourceFracCol = f32(uniforms.effectiveXSize[1]) *
                  (f32(dyC) / f32(uniforms.effectiveYSize[1]));

              let sourceNearestRow =
                  i32(min(f32(uniforms.outShape[1] - 1),
                  ${this.alignCorners?`floor(sourceFracRow + 0.5)`:`floor(sourceFracRow)`}));

              let sourceNearestCol =
                  i32(min(f32(uniforms.outShape[2] - 1),
                  ${this.alignCorners?`floor(sourceFracCol + 0.5)`:`floor(sourceFracCol)`}));

              if (r == sourceNearestRow && c == sourceNearestCol) {
                accumulator += getDy(b, dyR, dyC, d);
              }
            }
          }
          // End loop over dy

          setOutputAtIndex(index, accumulator);
        }
      }
    `}};function If(e){let{inputs:t,backend:n,attrs:r}=e,{images:i,dy:a}=t,{alignCorners:o}=r,[,s,c]=i.shape,[,l,u]=a.shape,d=[o&&l>1?s-1:s,o&&u>1?c-1:c],f=[o&&l>1?l-1:l,o&&u>1?u-1:u],p=d[0]/f[0],m=d[1]/f[1],h=1/p,g=1/m,_=Math.ceil(h)*2+2,v=Math.ceil(g)*2+2,y=new Ff(i.shape,o),b=[{type:`int32`,data:d},{type:`int32`,data:f},{type:`float32`,data:[h]},{type:`float32`,data:[g]},{type:`int32`,data:[_]},{type:`int32`,data:[v]}];return n.runWebGPUProgram(y,[a],a.dtype,b)}var Lf={kernelName:Re,backendName:`webgpu`,kernelFunc:If},Rf=class{constructor(e){this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.uniforms=` axis : vec4<i32>,`,this.shaderKey=`reverse`}getUserCode(){return`
      
      // Using uniform variables as judging conditions, so the function has
      // coherent execution within all threads.
      fn getReverseCoords(coords : vec4<i32>) -> vec4<i32> {
        var reverseCoords = coords;
        if (uniforms.axis[0] == 1) {
          reverseCoords[0] = uniforms.xShape[0] - coords[0] - 1;
        }
        if (uniforms.axis[1] == 1) {
          reverseCoords[1] = uniforms.xShape[1] - coords[1] - 1;
        }
        if (uniforms.axis[2] == 1) {
          reverseCoords[2] = uniforms.xShape[2] - coords[2] - 1;
        }
        if (uniforms.axis[3] == 1) {
          reverseCoords[3] = uniforms.xShape[3] - coords[3] - 1;
        }

        return reverseCoords;
      }
    
      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let reverseCoords = getReverseCoords(coords);
          setOutputAtIndex(index, getX(reverseCoords[0],
              reverseCoords[1], reverseCoords[2], reverseCoords[3]));
        }
      }
    `}};function zf(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{dims:a}=r,o=i.shape.length;if(o===0)return Y({inputs:{x:i},backend:n});let s=i.shape,c=[1,1,1,1];s.forEach((e,t)=>{let n=t+4-o;c[n]=e});let l=An(a,i.shape),u=[0,0,0,0];l.forEach(e=>{let t=e+4-o;u[t]=1});let d=[{type:`int32`,data:u}],f=J({inputs:{x:i},backend:n,attrs:{shape:c}}),p=new Rf(c),m=n.runWebGPUProgram(p,[f],f.dtype,d);n.disposeData(f.dataId);let h=J({inputs:{x:m},backend:n,attrs:{shape:s}});return n.disposeData(m.dataId),h}var Bf={kernelName:ie,backendName:`webgpu`,kernelFunc:zf},Vf=class{constructor(e,t){this.outputShape=[],this.variableNames=[`x`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.uniforms=`centerX : f32, centerY : f32, sinRadians : f32,
          cosRadians : f32,`,this.shaderKey=`rotate`,this.outputShape=e,typeof t==`number`?(this.uniforms+=` fillValue : f32,`,this.fillSnippet=`var outputValue = uniforms.fillValue;`,this.shaderKey+=`_float`):(this.uniforms+=` fillValue : vec3<f32>,`,this.fillSnippet=`var outputValue = uniforms.fillValue[coords[3]];`,this.shaderKey+=`_vec3`)}getUserCode(){return`
        ${z(`index`)} {
          if (index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            let coordXFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.cosRadians - (f32(coords[1]) - uniforms.centerY) *
                uniforms.sinRadians;
            let coordYFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.sinRadians + (f32(coords[1]) - uniforms.centerY) *
                uniforms.cosRadians;
            let coordX = i32(round(coordXFloat + uniforms.centerX));
            let coordY = i32(round(coordYFloat + uniforms.centerY));
            ${this.fillSnippet}
            if(coordX >= 0 && coordX < uniforms.xShape[2] && coordY >= 0 &&
                coordY < uniforms.xShape[1]) {
              outputValue = getX(coords[0], coordY, coordX, coords[3]);
            }
            setOutputAtIndex(index, outputValue);
          }
        }
      `}},Hf={kernelName:me,backendName:`webgpu`,kernelFunc:({inputs:e,attrs:t,backend:n})=>{let{image:r}=e,{radians:i,fillValue:a,center:o}=t,s=n,c=new Vf(r.shape,a),[l,u]=He(o,r.shape[1],r.shape[2]),d=[{type:`float32`,data:[l]},{type:`float32`,data:[u]},{type:`float32`,data:[Math.sin(i)]},{type:`float32`,data:[Math.cos(i)]}];return typeof a==`number`?d.push({type:`float32`,data:[Number.parseFloat(a.toFixed(2))]}):d.push({type:`float32`,data:a}),s.runWebGPUProgram(c,[r],r.dtype,d)}},Uf=X({opType:W.ROUND}),Wf={kernelName:Te,backendName:`webgpu`,kernelFunc:Uf},Gf=X({opType:W.RSQRT,cpuKernelImpl:No}),Kf={kernelName:Ke,backendName:`webgpu`,kernelFunc:Gf},qf=class{constructor(e,t,n,r,i,a,o,s=!0){this.variableNames=[`updates`,`indices`],this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=a,this.type=o,this.sumDupeIndices=s,this.dispatchLayout=V(e),this.dispatch=B(this.dispatchLayout,e,this.workgroupSize),this.sliceDimGreaterThanOne=t>1,this.shaderKey=`scatter_${n}_${r}_${this.sliceDimGreaterThanOne}_${o}_${s}_${i.length}`;let c=L(i.length);this.uniforms=`sliceDim : i32, strides: ${c}, updatesSize: i32,`,this.updatesRank=r,this.indicesRank=n}getUserCode(){let e=``;this.indicesRank===1?e=`coords[0]`:this.indicesRank===2&&(e=`coords[0], j`);let t=`getIndices(${e})`,n=this.sliceDimGreaterThanOne?`uniforms.strides[j]`:`uniforms.strides`,r=``,i=``;this.dispatchLayout.x.length===1?(r=`flattenedIndex`,i=`
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> i32 {
        return index;
      }
      `):this.dispatchLayout.x.length===2&&(r=`vec2<i32>(flattenedIndex, coords[1])`,i=`
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> vec2<i32> {
        // N.B. |updates| could be a scalar tensor, conceptually representing a
        // 2D tensor with all values equal to that. By design, its size must be
        // the same as |outShape[1]| in one dimension, and |indicesShape[0]|
        // gives the other.
        let sliceSize = uniforms.outShape[1];
        let d0 = index / sliceSize;
        let d1 = index - d0 * sliceSize;
        return vec2<i32>(d0, d1);
      }
      `);let a=`getUpdates(${Array.from({length:this.updatesRank},(e,t)=>`coords[${t}]`).join(`, `)})`;return`
    ${i}
      ${z(`index`)} {
        if (index < uniforms.updatesSize) {
          let coords = getUpdatesCoordsFromFlatIndex(index);
          var flattenedIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexInside = i32(round(${t}));
            flattenedIndex = flattenedIndex + indexInside * ${n};
          }
          let updateValue =
              ${Yr(this.type)}(${a});
          let flatIndex = getOutputIndexFromCoords(${r});

          ${this.sumDupeIndices?F(`&result[flatIndex]`,`updateValue`,this.type):`atomicStore(&result[flatIndex], bitcast<i32>(updateValue));`}
        }
      }`}};function Jf(e){let{inputs:t,backend:n,attrs:r}=e,{indices:i,updates:a}=t,{shape:s}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=o(a,i,s),p=[f/u,u];if(f===0)return n.makeTensorInfo(s,i.dtype);let m=J({inputs:{x:i},backend:n,attrs:{shape:[l,c]}}),h=J({inputs:{x:a},backend:n,attrs:{shape:[l,u]}}),g=h.dtype,_=q({backend:n,attrs:{shape:p,value:0,dtype:g}}),v=M(h.shape),y=[{type:`int32`,data:[c]},{type:`int32`,data:d},{type:`int32`,data:[v]}],b=new qf(h.shape,c,m.shape.length,h.shape.length,d,p,g),x=n.runWebGPUProgram(b,[h,m],g,y,_),S=J({inputs:{x},backend:n,attrs:{shape:s}});return n.disposeData(m.dataId),n.disposeData(h.dataId),n.disposeData(x.dataId),S}var Yf={kernelName:Tr,backendName:`webgpu`,kernelFunc:Jf},Xf=class{constructor(e,t){this.outputShape=[],this.variableNames=[`sortedSequence`,`values`],this.uniforms=`numInputs : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.side=t,this.shaderKey=`search_sorted_${t}`}getUserCode(){return`
      fn findBound(batch: i32, value: f32) -> i32 {
        var left = i32(0);
        var right = uniforms.numInputs;
        while (left < right) {
          var mid = (left + right) / 2;
          if (getSortedSequence(batch, mid) ${this.side===`left`?`<`:`<=`} value) {
            left = mid + 1;
          } else {
            right = mid;
          }
        }
        return right;
      }

      ${z(`index`)} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let value = getValuesByOutputIndex(index);
          setOutputAtIndexI32(index, findBound(coords[0], value));
        }
      }
    `}};function Zf(e){let{inputs:t,backend:n,attrs:r}=e,{sortedSequence:i,values:a}=t,{side:o}=r,s=new Xf([a.shape[0],a.shape[1]],o),c=[{type:`int32`,data:[i.shape[1]]}];return n.runWebGPUProgram(s,[i,a],`int32`,c)}var Qf={kernelName:m,backendName:`webgpu`,kernelFunc:Zf},$f=class{constructor(e,t,n){this.variableNames=[`c`,`a`,`b`],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.cRank=e,this.rank=n,this.shaderKey=`select`}getUserCode(){let e,t;if(this.rank>4)throw Error(`Where for rank ${this.rank} is not yet supported`);if(this.rank===1)t=`resRC`,e=`resRC`;else{let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],r=[],i=[];for(let e=0;e<this.outputShape.length;e++)i.push(`${n[e]}`),e<this.cRank&&r.push(`${n[e]}`);e=r.join(),t=i.join()}return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let cVal = getC(${e});
          if (cVal >= 1.0) {
            setOutputAtIndex(index, getA(${t}));
          } else {
            setOutputAtIndex(index, getB(${t}));
          }
        }
      }
    `}};function ep(e){let{inputs:t,backend:n}=e,{condition:r,t:i,e:a}=t,o=new $f(r.shape.length,i.shape,i.shape.length);return n.runWebGPUProgram(o,[r,i,a],f(i.dtype,a.dtype))}var tp={kernelName:lt,backendName:`webgpu`,kernelFunc:ep},np=X({opType:W.SELU}),rp={kernelName:ue,backendName:`webgpu`,kernelFunc:np},ip=X({opType:W.SIGMOID}),ap={kernelName:at,backendName:`webgpu`,kernelFunc:ip},op=X({opType:W.SIGN}),sp={kernelName:mt,backendName:`webgpu`,kernelFunc:op},cp={kernelName:`Sin`,backendName:`webgpu`,kernelFunc:X({opType:W.SIN})},lp=X({opType:W.SINH}),up={kernelName:be,backendName:`webgpu`,kernelFunc:lp},dp=X({opType:W.SOFTPLUS}),fp={kernelName:Tt,backendName:`webgpu`,kernelFunc:dp},pp=class{constructor(e,t,n,r,i,a){this.variableNames=[`x`],this.outputShape=[],this.uniforms=``,this.workgroupSize=[64,1,1],this.size=!0;let o=Array(r.length);for(let e=0;e<o.length;e++)o[e]=r[i[e]];this.outputShape=o,this.newDim=i,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.xShape=e,this.paddedXShape=t,this.uniforms+=`reshapedPaddedXShape : ${L(r.length)}, paddedXShapeStrides : ${L(a)}, `,n.map((e,t)=>{this.uniforms+=` pad${t} : vec2<i32>,`}),this.shaderKey=`spaceToBatchND_${i}`}getUserCode(){let e=L(this.outputShape.length),t=ts(this.newDim);return`
      ${Hr(this.paddedXShape,`PaddedX`)}
      ${z(`index`)} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let switchedIndex = getIndexFromCoords${this.outputShape.length}D(${e}(${t}), uniforms.reshapedPaddedXShape);
          let paddedCoords = getPaddedXCoordsFromIndex(switchedIndex);
          ${lf(this.xShape,!0)}
        }
      }
    `}},mp={kernelName:yt,backendName:`webgpu`,kernelFunc:e=>{let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{blockShape:a,paddings:o}=r;k(i.shape.length<=4,()=>`spaceToBatchND for rank > 4 with a WebGPU backend not implemented yet`);let s=a.reduce((e,t)=>e*t),c=[[0,0]];c.push(...o);for(let e=1+a.length;e<i.shape.length;++e)c.push([0,0]);let l=c.map((e,t)=>e[0]+i.shape[t]+e[1]),u=_e(l,a,s,!1),d=Wn(u.length,a.length,!1),f=Et(l,a,s,!1),p=We(l),m=new pp(i.shape,l,c,u,d,p.length),h=[{type:`int32`,data:u},{type:`int32`,data:p}];c.map(e=>h.push({type:`int32`,data:[e[0],e[1]]}));let g=n.runWebGPUProgram(m,[i],i.dtype,h),_=J({inputs:{x:g},backend:n,attrs:{shape:f}});return n.disposeData(g.dataId),_}},hp=class{constructor(e,t,n){this.variableNames=[`input`,`indices`,`segmentIds`],this.outputShape=[],this.uniforms=`segmentSize : i32, sparseSize : i32,`,this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=e,this.type=n,this.dispatchLayout=V([t]),this.dispatch=B(this.dispatchLayout,[t],this.workgroupSize),this.shaderKey=`sparseSegmentSum`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.sparseSize) {
        let indexInSegmentIds = index / uniforms.segmentSize;
        let indexInSegment = index % uniforms.segmentSize;
        let indexInInput = indices[indexInSegmentIds];
        let segmentId = segmentIds[indexInSegmentIds];

        let value = input[indexInInput * uniforms.segmentSize + indexInSegment];
        let outIndex = segmentId * uniforms.segmentSize + indexInSegment;
        ${F(`&result[outIndex]`,`value`,this.type)}
      }
    }
  `}},gp=class{constructor(e,t){this.variableNames=[`segmentIds`],this.outputShape=[],this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=[e],this.dispatchLayout=V(t),this.dispatch=B(this.dispatchLayout,t,this.workgroupSize),this.shaderKey=`sparseSegmentIdCountProgram`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.segmentIdsShape) {
        let segmentId = segmentIds[index];
        ${F(`&result[segmentId]`,`1`,`int32`)}
      }
    }
  `}},_p=class{constructor(e,t){this.variableNames=[`segmentSum`,`sameSegmentIdCount`],this.outputShape=[],this.uniforms=`segmentSize : i32`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.type=t,this.dispatchLayout=V(e),this.dispatch=B(this.dispatchLayout,e,this.workgroupSize),this.shaderKey=`sparseSegmentMean`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.size) {
        let segmentId = index / uniforms.segmentSize;
        let count = sameSegmentIdCount[segmentId];
        if (count != 0) {
          ${this.type===`float32`?`setOutputAtIndex(index, segmentSum[index] / f32(count));`:`setOutputAtIndexI32(index, segmentSum[index] / count);`}
        }
      }
    }
  `}};function vp(e,t,n,r=!1,i){let a=M(e.shape)/e.shape[0],o=e.dtype,s=M(t.shape),c=i.readSync(n.dataId),l=s>0?c[s-1]+1:0,u,d=e.shape.slice();d[0]=l;let f=s*a,p=q({backend:i,attrs:{shape:d,value:0,dtype:o}});u=new hp(d,f,o);let m=[{type:`int32`,data:[a]},{type:`int32`,data:[f]}],h=i.runWebGPUProgram(u,[e,t,n],o,m,p);if(r)return h;let g=q({backend:i,attrs:{shape:[l],value:0,dtype:`int32`}});u=new gp(l,n.shape);let _=i.runWebGPUProgram(u,[n],`int32`,null,g),v=q({backend:i,attrs:{shape:d,value:0,dtype:o}});u=new _p(d,o),m=[{type:`int32`,data:[a]}];let y=i.runWebGPUProgram(u,[h,_],o,m,v);return i.disposeData(h.dataId),i.disposeData(_.dataId),y}function yp(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;return vp(r,i,a,!1,n)}var bp={kernelName:i,backendName:`webgpu`,kernelFunc:yp};function xp(e){let{inputs:t,backend:n}=e,{data:r,indices:i,segmentIds:a}=t;return vp(r,i,a,!0,n)}var Sp={kernelName:Yt,backendName:`webgpu`,kernelFunc:xp},Cp=class{constructor(e,t){this.variableNames=[`A`],this.workgroupSize=[64,1,1],this.size=!0;let n=Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[r]*t[r];this.outputShape=n,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.rank=this.outputShape.length,this.shaderKey=`tile`}getUserCode(){let e=wp(this.rank,`uniforms.`);return`
      ${z(`index`)} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          setOutputAtIndex(index, getA(${e}));
        }
      }
    `}};function wp(e,t=``){if(e>=5)throw Error(`Tile for rank ${e} is not yet supported`);if(e===1)return`(resRC % ${t}aShape)`;let n=[`resRC.x`,`resRC.y`,`resRC.z`,`resRC.w`],r=[];for(let i=0;i<e;i++)r.push(`(${n[i]} % ${t}aShape[${i}])`);return r.join()}function Tp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{reps:a}=r;if(n.shouldExecuteOnCPU([i])||i.dtype===`string`||i.shape.length>=5){let e=n.readSync(i.dataId),t=i.dtype===`string`?e.map(e=>ye(e)):e,r=Bo(Oe(i.shape,i.dtype,t),a);return n.makeTensorInfo(r.shape,r.dtype,r.values)}let o=new Cp(i.shape,a);return n.runWebGPUProgram(o,[i],i.dtype)}var Ep={kernelName:jt,backendName:`webgpu`,kernelFunc:Tp};function Dp(e){let{inputs:t,backend:n,attrs:r}=e,{sparseIndices:i,sparseValues:a,defaultValue:s}=t,{outputShape:c}=r,{sliceRank:l,numUpdates:u,sliceSize:d,strides:f,outputSize:p}=o(a,i,c);if(a.dtype===`string`){let e=Po(n.bufferSync(i),n.bufferSync(a),c,p,d,u,l,f,ye(n.readSync(s.dataId)[0]),!1);return n.makeTensorInfo(c,e.dtype,e.values)}let m=[p/d,d],h=J({inputs:{x:i},backend:n,attrs:{shape:[u,l]}}),g=a.shape.length?J({inputs:{x:a},backend:n,attrs:{shape:[u,d]}}):Y({inputs:{x:a},backend:n}),_=g.dtype,v=n.makeTensorInfo([],_,Xt(1,_)),y=J({inputs:{x:s},backend:n,attrs:{shape:Array(m.length).fill(1)}}),b=Tp({inputs:{x:y},backend:n,attrs:{reps:m}}),x=M([u,d]),S=[{type:`int32`,data:[l]},{type:`int32`,data:f},{type:`int32`,data:[x]}];switch(u){case 0:break;case 1:{let e=new qf([u,d],l,h.shape.length,g.shape.length,f,m,_,!1);n.runWebGPUProgram(e,[g,h],_,S,b)}break;default:{let e=new qf([u,d],l,h.shape.length,v.shape.length,f,m,_,!1);n.runWebGPUProgram(e,[v,h],_,S,b)}{let e=new qf([u,d],l,h.shape.length,g.shape.length,f,m,_);n.runWebGPUProgram(e,[g,h],_,S,b)}}let C=J({inputs:{x:b},backend:n,attrs:{shape:c}});return n.disposeData(h.dataId),n.disposeData(g.dataId),n.disposeData(y.dataId),n.disposeData(v.dataId),n.disposeData(b.dataId),C}var Op={kernelName:$n,backendName:`webgpu`,kernelFunc:Dp};function kp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{numOrSizeSplits:a,axis:o}=r,s=An(o,i.shape)[0],c=nt(i,a,s),l=i.shape.length,u=Array(l).fill(0),d=i.shape.slice();return c.map(e=>{let t=[...d];t[s]=e;let r=qs({inputs:{x:i},backend:n,attrs:{begin:u,size:t}});return u[s]+=e,r})}var Ap={kernelName:kn,backendName:`webgpu`,kernelFunc:kp},jp=X({opType:W.SQRT}),Mp={kernelName:Gn,backendName:`webgpu`,kernelFunc:jp},Np={kernelName:pn,backendName:`webgpu`,kernelFunc:({inputs:e,backend:t})=>{let{x:n}=e,r=t,i=new so(n.shape,W.SQUARE);return r.runWebGPUProgram(i,[n],n.dtype)}},Pp=Z({opType:U.SQUARED_DIFFERENCE}),Fp={kernelName:Ft,backendName:`webgpu`,kernelFunc:Pp};function Ip({inputs:e,attrs:t,backend:n}){let{x:r}=e,i=new so(r.shape,W.STEP,`stepAlpha : f32,`),a=[{type:`float32`,data:[t.alpha]}];return n.runWebGPUProgram(i,[r],r.dtype,a)}var Lp={kernelName:Yn,backendName:`webgpu`,kernelFunc:Ip},Rp=class{constructor(e){this.variableNames=[`x`],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]);let t=L(this.outputShape.length);this.uniforms=`begin : ${t},  strides : ${t}, `,this.shaderKey=`stridedSlice`}getUserCode(){let e=this.outputShape.length,t=``;if(e===1)t=`coords * uniforms.strides + uniforms.begin`;else{let e=0;t=this.outputShape.map((t,n)=>(e++,this.outputShape.length===1?`coords * uniforms.strides[${n}] + uniforms.begin[${n}]`:`coords[${e-1}] * uniforms.strides[${n}] + uniforms.begin[${n}]`)).join(`,`)}return`
       ${z(`index`)} {
         if (index < uniforms.size) {
           let coords = getCoordsFromIndex(index);
           setOutputAtIndex(index, getX(${t}));
         }
       }
     `}};function zp(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{begin:a,end:o,strides:s,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:f}=r,{finalShapeSparse:p,finalShape:m,isIdentity:h,sliceDim0:g,isSimpleSlice:_,begin:v,end:y,strides:b}=Xn(i.shape,a,o,s,c,l,u,d,f),x;if(h)x=J({inputs:{x:i},backend:n,attrs:{shape:m}});else if(g||_){k(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);let e=mn(v,y,b),t=qs({inputs:{x:i},backend:n,attrs:{begin:v,size:e}});x=J({inputs:{x:t},backend:n,attrs:{shape:m}}),n.disposeData(t.dataId)}else if(n.shouldExecuteOnCPU([i])){let e=n.readSync(i.dataId),t=Lo(p,Oe(i.shape,i.dtype,e),b,v);x=n.makeTensorInfo(m,i.dtype,t.values)}else{let e=new Rp(p),t=[{type:`int32`,data:v},{type:`int32`,data:b}],r=n.runWebGPUProgram(e,[i],i.dtype,t);x=J({inputs:{x:r},backend:n,attrs:{shape:m}}),n.disposeData(r.dataId)}return x}var Bp={kernelName:Ut,backendName:`webgpu`,kernelFunc:zp};function Vp(e){let{inputs:t,backend:n,attrs:r}=e,{separator:i,nGramWidths:a,leftPad:o,rightPad:s,padWidth:c,preserveShortSequences:l}=r,{data:u,dataSplits:d}=t,[f,p]=Ro(n.readSync(u.dataId),n.readSync(d.dataId),i,a,o,s,c,l);return[n.makeTensorInfo([f.length],`string`,f),n.makeTensorInfo(d.shape,`int32`,p)]}var Hp={kernelName:wn,backendName:`webgpu`,kernelFunc:Vp},Up={kernelName:`Sub`,backendName:`webgpu`,kernelFunc:Z({opType:U.SUB,cpuKernelImpl:zo,supportsComplex:!0})},Wp={kernelName:`Tan`,backendName:`webgpu`,kernelFunc:X({opType:W.TAN})},Gp=X({opType:W.TANH}),Kp={kernelName:un,backendName:`webgpu`,kernelFunc:Gp};function qp(e){let{inputs:t,backend:n,attrs:r}=e,{tensor:i,indices:a,updates:s}=t,{}=r,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:f}=o(s,a,i.shape),p=[f/u,u];if(f===0)return n.makeTensorInfo(i.shape,a.dtype);let m=[],h=J({inputs:{x:a},backend:n,attrs:{shape:[l,c]}});m.push(h);let g=J({inputs:{x:s},backend:n,attrs:{shape:[l,u]}});m.push(g);let _=J({inputs:{x:i},backend:n,attrs:{shape:p}});m.push(_);let v=Tp({inputs:{x:_},backend:n,attrs:{reps:Array(p.length).fill(1)}}),y=new qf([l,u],c,h.shape.length,g.shape.length,d,p,i.dtype,!1),b=M([l,u]),x=[{type:`int32`,data:[c]},{type:`int32`,data:d},{type:`int32`,data:[b]}],S=n.runWebGPUProgram(y,[g,h],_.dtype,x,v);m.push(S);let C=J({inputs:{x:S},backend:n,attrs:{shape:i.shape}});return m.forEach(e=>n.disposeData(e.dataId)),C}var Jp={kernelName:on,backendName:`webgpu`,kernelFunc:qp},Yp=class{constructor(e){this.variableNames=[`x`,`indices`],this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.uniforms=`inputSize : i32, firstPass : i32, negativeInf : f32,
        dir : i32, inc : i32,`,this.shaderKey=`swap`}getUserCode(){return`
        ${z(`index`)} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // We compare elements pair-wise within a group of size 2 * inc.
            // The comparing rule for each group alternates between ascending
            // and descending. Within each group, we compare each pair at
            // positions i and i+inc. To decide whether an element at position i
            // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
            // inc, it is in the first half of the group, we denote it as x0,
            // otherwise we denote it as x1.
            // For example, as shown in the Bitonic top K paper referenced
            // above, Figure5(a) shows that element[1] is in the second half of
            // the group when group size is 2, but it is in the first half of
            // the group when group size is 4.
            let isFirstInPair = elemIdx % (2 * uniforms.inc) < uniforms.inc;
            var i = 0;
            if (isFirstInPair) {
              i = elemIdx;
            } else {
              i = elemIdx - uniforms.inc;
            }

            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }

            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.inc;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.inc));
            }

            var x0 = f32(0.0);
            var x1 = f32(0.0);
            if (i0 < uniforms.inputSize) {
              x0 = getX(batch, i0);
            } else {
              x0 = uniforms.negativeInf;
            }
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = uniforms.negativeInf;
            }

            let reverse = elemIdx % (2 * uniforms.dir) >= uniforms.dir;
            let isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
            if (reverse == isGreater) {
              // Elements in opposite order of direction
              let iTemp = i0;
              i0 = i1;
              i1 = iTemp;
            }
            if (isFirstInPair) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `}},Xp=class{constructor(e){this.variableNames=[`x`,`indices`],this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.uniforms=`inputSize : i32, firstPass : i32, k : i32,`,this.shaderKey=`merge`}getUserCode(){return`
        ${z(`index`)} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // The output size is half of the previous size.
            // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _
            // (k=4), we only need to output the indices at positions |, the
            // indices at positions _ can be thrown away, see Figure5(b) After
            // Phase 2 (Merge phase) in the Bitonic Top K paper referenced
            // above.
            // For example, the paper shows we only need to output the orange
            // bars. The output sequence should look like this | | | | | | | |.
            // Because the sequence is halved, to map the output index back to
            // the previous sequence to find the corresponding value, we need
            // to double the index. When we double the index, we basically
            // interpolate a position, so 2i looks like
            // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k
            // position of each 2k positions by - elemIdx % k. E.g. for output
            // at index 4,5,6,7, we want to get the corresponding element at
            // original index 8,9,10,11, for output at index 8,9,10,11,
            // we want to get the corresponding element at original index
            // 16,17,18,19, so on and so forth.

            var i = 0;
            if (elemIdx < uniforms.k) {
              i = elemIdx;
            } else {
              i = elemIdx * 2 - elemIdx % uniforms.k;
            }
            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }
            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.k;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.k));
            }

            let x0 = getX(batch, i0);
            var x1 = f32(0.0);
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = x0;
            }

            if (x0 >= x1) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `}};function Zp(e,t){t!==null&&e.disposeData(t.dataId)}function Qp(e){let t=1;for(;t<e;)t*=2;return t}function $p(e){let{inputs:t,backend:n,attrs:r}=e,{x:i}=t,{k:a,sorted:o}=r,s=i.shape,c=s[s.length-1];if(n.shouldExecuteOnCPU([i])){let[e,t]=Vo(n.readSync(i.dataId),s,i.dtype,a,o);return[n.makeTensorInfo(e.shape,e.dtype,e.values),n.makeTensorInfo(t.shape,t.dtype,t.values)]}if(a===0)return s[s.length-1]=0,[n.makeTensorInfo(s,i.dtype,[]),n.makeTensorInfo(s,`int32`,[])];if(c===1)return[i,q({attrs:{shape:s,dtype:`int32`,value:0},backend:n})];let l=M(s)/c,u=J({inputs:{x:i},attrs:{shape:[l,c]},backend:n}),d=Qp(a),f=Qp(c),p=null,m=()=>p===null?[u,u]:[u,p],h=(e,t,r)=>{let i=m(),a=new Yp(r),o=[{type:`int32`,data:[c]},{type:`int32`,data:[+(p===null)]},{type:`float32`,data:[-1/0]},{type:`int32`,data:[e]},{type:`int32`,data:[t]}],s=p;p=n.runWebGPUProgram(a,i,`int32`,o),Zp(n,s)};for(let e=1;e<d;e*=2){let t=e*2;for(let n=e;n>=1;n/=2)h(t,n,[l,f])}for(let e=f;e>d;e/=2){let t=m(),r=new Xp([l,e/2]),i=[{type:`int32`,data:[c]},{type:`int32`,data:[+(p===null)]},{type:`int32`,data:[d]}],a=p;p=n.runWebGPUProgram(r,t,`int32`,i),Zp(n,a);let o=d/2,s=o*2;for(let e=o;e>=1;e/=2)h(s,e,p.shape)}let g=p;p=qs({inputs:{x:p},backend:n,attrs:{begin:0,size:[l,a]}}),Zp(n,g);let _=Pu({inputs:{x:u,indices:p},backend:n,attrs:{axis:1,batchDims:1}});Zp(n,u);let v=s.slice(0,-1);v.push(a),g=p,p=J({inputs:{x:p},attrs:{shape:v},backend:n}),Zp(n,g);let y=_;return _=J({inputs:{x:_},attrs:{shape:v},backend:n}),Zp(n,y),[_,p]}var em={kernelName:sr,backendName:`webgpu`,kernelFunc:$p},tm=class{constructor(e){this.variableNames=[`Image`,`Transforms`],this.uniforms=`interpolationModeId : i32, fillModeId : i32, fillValue : f32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=V(this.outputShape),this.dispatch=B(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey=`transform`}getUserCode(){return`
          fn mapCoord(outCoord : f32, len : f32) -> f32{
            var inCoord = outCoord;
            if(uniforms.fillModeId == 2) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  if (inCoord < sz2) {
                    inCoord = sz2 * f32(i32(f32(-inCoord / sz2))) +
                    inCoord;
                  }
                  if (inCoord < -len) {
                    inCoord = inCoord + sz2;
                  } else {
                    inCoord = -inCoord - 1.0;
                  }
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  inCoord = inCoord - sz2 * f32(i32(f32(inCoord / sz2)));
                  if (inCoord >= len) {
                    inCoord = sz2 - inCoord - 1.0;
                  }
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 3) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord + len * (f32(i32(f32(-inCoord / sz))) + 1.0);
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord - len * f32(i32(f32(inCoord / sz)));
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 4) {
              return clamp(outCoord, 0.0, len - 1.0);
            }
            return outCoord;
          }
          fn readWithFillValue(batch : i32, coordY : i32, coordX : i32,
            channel : i32) -> f32 {
            var outputValue : f32;
            if (0 <= coordY && coordY < uniforms.imageShape[1] && 0 <= coordX && coordX < uniforms.imageShape[2]) {
                outputValue = getImage(batch, coordY, coordX, channel);
            } else {
              outputValue = uniforms.fillValue;
            }
            return outputValue;
          }

          ${z(`index`)} {
            if (index < uniforms.size) {
              let coords = getCoordsFromIndex(index);
              var outputValue : f32;
              let batch = coords[0];
              let x = coords[2];
              let y = coords[1];
              let channel = coords[3];
              let xf = f32(x);
              let yf = f32(y);
              let a1 = getTransforms(batch, 0);
              let a2 = getTransforms(batch, 1);
              let a3 = getTransforms(batch, 2);
              let b1 = getTransforms(batch, 3);
              let b2 = getTransforms(batch, 4);
              let b3 = getTransforms(batch, 5);
              let c1 = getTransforms(batch, 6);
              let c2 = getTransforms(batch, 7);
              let projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = uniforms.fillValue;
              } else {
                let inX = (a1 * xf + a2 * yf + a3) / projection;
                let inY = (b1 * xf + b2 * yf + b3) / projection;
                let mapX = mapCoord(inX, f32(uniforms.imageShape[2]));
                let mapY = mapCoord(inY, f32(uniforms.imageShape[1]));

                if (uniforms.interpolationModeId == 1) {
                  let coordY = i32(round(mapY));
                  let coordX = i32(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  let yFloor = floor(mapY);
                  let xFloor = floor(mapX);
                  let yCeil = yFloor + 1.0;
                  let xCeil = xFloor + 1.0;
                  let valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yFloor), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yFloor), i32(xCeil), channel);
                  let valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yCeil), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yCeil), i32(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutputAtIndex(index, outputValue);
            }
          }
        `}};function nm(e){let{inputs:t,backend:n,attrs:r}=e,{image:i,transforms:a}=t,{interpolation:o,fillMode:s,fillValue:c,outputShape:l}=r,[u,d,f,p]=i.shape,[m,h]=l??[d,f],g=new tm([u,m,h,p]),_=o===`nearest`?1:2,v;switch(s){case`constant`:v=1;break;case`reflect`:v=2;break;case`wrap`:v=3;break;case`nearest`:v=4;break;default:v=1}let y=[{type:`int32`,data:[_]},{type:`int32`,data:[v]},{type:`float32`,data:[c]}];return n.runWebGPUProgram(g,[i,a],`float32`,y)}var rm={kernelName:vr,backendName:`webgpu`,kernelFunc:nm};function im(e){let{inputs:t,backend:n,attrs:r}=e,{value:i}=t,{axis:a}=r;a<0&&(a+=i.shape.length);let o=i,s=o.shape.length,c=i.shape[a],l=Array(s-1),u=0;for(let e=0;e<s;e++)e!==a&&(l[u++]=o.shape[e]);let d=[],f=Array(s).fill(0),p=o.shape.slice();p[a]=1;let m=Array(c);for(let e=0;e<m.length;e++){f[a]=e;let t=qs({inputs:{x:o},backend:n,attrs:{begin:f,size:p}}),r=J({inputs:{x:t},backend:n,attrs:{shape:l}});m[e]=r,d.push(t)}return d.forEach(e=>n.disposeData(e.dataId)),m}var am={kernelName:Je,backendName:`webgpu`,kernelFunc:im},om=class{constructor(e,t,n){if(this.outputShape=[],this.variableNames=[`x`,`segmentIds`],this.uniforms=`numSegments : i32, xSize: i32,`,this.workgroupSize=[64,1,1],this.atomic=!0,this.outputShape=t,this.dispatchLayout=V(e),this.dispatch=B(this.dispatchLayout,e,this.workgroupSize),n!==`float32`&&n!==`int32`)throw Error(`UnsortedSegmentSum only supports float32 and int32
              types, does not support ${n} type.`);this.type=n,this.shaderKey=`unsortedSegmentSum`}getUserCode(){return`
    ${z(`index`)} {
      if (index < uniforms.xSize) {
        let coords = getXCoordsFromIndex(index);
        let b = coords[0];
        let inCol = coords[1];

        let segmentId = i32(getSegmentIds(inCol));
        if (segmentId >= 0) {
          let flatIndex = b * uniforms.numSegments + segmentId % uniforms.numSegments;
          let value = getX(b, inCol);

          ${F(`&result[flatIndex]`,`value`,this.type)}
        }
      }
    }
  `}};function sm(e){let{inputs:t,backend:n,attrs:r}=e,{x:i,segmentIds:a}=t,{numSegments:o}=r,s=i.shape.length,c=[],l=0,u=nr([l],s),d=i;u!=null&&(d=Q({inputs:{x:i},backend:n,attrs:{perm:u}}),c.push(d),l=Nn(1,s)[0]);let f=b(d.shape,l,o),p=M([d.shape[l]]),m=J({inputs:{x:d},backend:n,attrs:{shape:[-1,p]}});c.push(m);let h=i.dtype,g=[m.shape[0],o],_=q({backend:n,attrs:{shape:g,value:0,dtype:h}}),v=new om(m.shape,g,h),y=[{type:`int32`,data:[o]},{type:`int32`,data:[M(m.shape)]}],x=n.runWebGPUProgram(v,[m,a],h,y,_),S=J({inputs:{x},backend:n,attrs:{shape:f}});c.push(x);let C=S;if(u!=null){c.push(S);let e=qn(u);C=Q({inputs:{x:C},backend:n,attrs:{perm:e}})}return c.forEach(e=>n.disposeData(e.dataId)),C}var cm=[to,Wo,Ko,Jo,Yo,Qo,os,cs,ds,ps,hs,_s,ys,xs,Cs,Ns,Fs,zs,Vs,Us,Ys,ec,rc,uc,fc,gc,oo,bc,Dc,Lc,Uc,qc,Xc,Qc,el,tl,rl,al,fl,ml,gl,yl,Ol,Al,wl,Nl,Il,Bl,Hl,Gl,Zl,Ql,$l,tu,nu,ru,au,su,du,Za,pu,yu,hu,_u,wu,Eu,Ou,ju,Fu,Lu,zu,io,Vu,Cc,Uu,Gu,qu,Yu,Zu,$u,nd,ad,rd,sd,ld,dd,gd,yd,Os,xd,Cd,jd,Td,kd,Nd,As,Fd,Ld,zd,Bd,Kd,ql,Jd,Xd,Qd,ac,tf,of,cf,df,ff,mf,gf,_f,sc,yf,xf,Cf,Tf,Qa,Of,jf,Pf,Lf,Bf,Hf,Wf,Kf,Yf,Qf,tp,rp,ap,sp,cp,up,Js,Lp,Bp,Hp,Wd,fp,mp,bp,Sp,Op,Ap,Mp,Np,Fp,Up,Yl,Wp,Kp,Jp,Ep,em,rm,ns,am,{kernelName:_,backendName:`webgpu`,kernelFunc:sm},rf];for(let e of cm)Pt(e);export{ui as WebGPUBackend,$r as webgpu_util};