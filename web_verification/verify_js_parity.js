const fs=require('fs');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
function forward(raw,m){
 let a=raw.map((v,i)=>(v-m.scaler.mean[i])/m.scaler.scale[i]);
 for(let l=0;l<m.layers.length;l++){
  const layer=m.layers[l],last=l===m.layers.length-1;
  a=layer.b.map((bias,j)=>{
   let value=bias;
   for(let i=0;i<a.length;i++) value+=a[i]*layer.W[i][j];
   return last ? (value>=0?1/(1+Math.exp(-value)):Math.exp(value)/(1+Math.exp(value))) : Math.max(0,value);
  });
 }
 return a[0];
}
let result={};
for(let [name,d] of Object.entries(data)){
 let errors=[], max=0,mean=0, nOver1e6=0;
 for(let i=0;i<d.rows.length;i++){
  let value=forward(d.rows[i],d.model), diff=Math.abs(value-d.reference[i]);
  if(!Number.isFinite(value))throw Error('Nonfinite '+name+' '+i);
  max=Math.max(max,diff);mean+=diff;
  if(diff>1e-6)nOver1e6++;
  if(i<5)errors.push({row:i,js:value,reference:d.reference[i],abs_error:diff});
 }
 result[name]={n:d.rows.length,max_abs_error:max,mean_abs_error:mean/d.rows.length,count_exceed_1e6:nOver1e6,samples:errors};
}
fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));