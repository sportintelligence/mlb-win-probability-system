const fs=require('fs'); const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
function forward(raw,m){let a=raw.map((v,i)=>(v-m.scaler.mean[i])/m.scaler.scale[i]);
 for(let l=0;l<m.layers.length;l++){const layer=m.layers[l],last=l===m.layers.length-1;
  a=layer.b.map((bias,j)=>{let v=bias;for(let i=0;i<a.length;i++)v+=a[i]*layer.W[i][j];return last?(v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v))):Math.max(0,v);});}
 return a[0];}
for(const [name,d] of Object.entries(data)){
 for(let w=0;w<3;w++) for(const r of d.rows) forward(r,d.model); // warm-up
 const reps=5, times=[];
 for(let k=0;k<reps;k++){const t0=process.hrtime.bigint(); for(const r of d.rows) forward(r,d.model); const t1=process.hrtime.bigint(); times.push(Number(t1-t0)/1e6/d.rows.length);}
 times.sort((a,b)=>a-b); console.log(name, 'rows',d.rows.length,'median ms per prediction',times[2].toFixed(4));
}
