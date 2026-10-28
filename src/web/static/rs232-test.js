"use strict";
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function api(url,options={}){const response=await fetch(url,{cache:"no-store",...options});if(!response.ok){let error=`HTTP ${response.status}`;try{error=(await response.json()).detail||error}catch(_){}throw new Error(error)}return response.json()}
const post=(url,data={})=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
const connection=()=>({port:$("serial-port").value,baudrate:Number($("serial-baud").value)});
const values=()=>[1,2,3,4,5,6].map(i=>Number($(`channel-${i}`).value));
function render(data){
  $("serial-status").textContent=data.status;
  $("serial-message").textContent=data.message;
  data.notes.forEach((note,index)=>{$(`channel-note-${index+1}`).textContent=note});
  $("serial-history").replaceChildren();
  data.history.forEach(entry=>{const tr=document.createElement("tr");entry.forEach(cell=>{const td=document.createElement("td");td.textContent=cell;tr.append(td)});$("serial-history").append(tr)});
}
async function action(url,body={}){try{render(await post(url,body))}catch(error){$("serial-message").textContent=`⚠ ${error.message}`}}
document.addEventListener("DOMContentLoaded",async()=>{
  try{const data=await api("/api/rs232-test/bootstrap");$("serial-port").value=data.port;$("serial-baud").value=String(data.baudrate);$("serial-ports").innerHTML=data.ports.map(port=>`<option value="${esc(port)}">`).join("");
    $("serial-channels").innerHTML=Array.from({length:data.channels},(_,index)=>`<div class="channel-card"><h3>CH${index+1}</h3><div class="value" id="channel-value-${index+1}">0%</div><input id="channel-${index+1}" type="range" min="0" max="100" step="1" value="0"><button data-channel="${index+1}">CH${index+1} 적용</button><p id="channel-note-${index+1}" class="hint">최근 ACK: 미확인</p><p id="channel-packet-${index+1}" class="hint"></p></div>`).join("");
    for(let i=1;i<=data.channels;i++){$(`channel-${i}`).addEventListener("input",()=>{const value=Number($(`channel-${i}`).value),output=Math.round(value*255/100);$(`channel-value-${i}`).textContent=`${value}%`;const bytes=`02 ${i.toString().charCodeAt(0).toString(16).toUpperCase().padStart(2,"0")} ${String(output).padStart(3,"0").split("").map(c=>c.charCodeAt(0).toString(16).toUpperCase()).join(" ")} 03`; $(`channel-packet-${i}`).textContent=`TX 예정: ${bytes}`});$(`channel-${i}`).dispatchEvent(new Event("input"))}
    document.querySelectorAll("[data-channel]").forEach(button=>button.addEventListener("click",()=>action("/api/rs232-test/apply",{...connection(),channel:Number(button.dataset.channel),values:[Number($(`channel-${button.dataset.channel}`).value)]})));
    render(data);
  }catch(error){$("serial-status").textContent=`시작 실패: ${error.message}`}
  $("serial-connect").addEventListener("click",()=>action("/api/rs232-test/connect",connection()));
  $("serial-disconnect").addEventListener("click",()=>action("/api/rs232-test/disconnect"));
  $("serial-apply-all").addEventListener("click",()=>action("/api/rs232-test/apply",{...connection(),values:values()}));
  $("serial-off").addEventListener("click",async()=>{try{const data=await post("/api/rs232-test/off",connection());render(data);data.reset_channels.forEach(channel=>{$(`channel-${channel}`).value=0;$(`channel-${channel}`).dispatchEvent(new Event("input"))})}catch(error){$("serial-message").textContent=error.message}});
  $("clear-serial-history").addEventListener("click",()=>action("/api/rs232-test/history/clear"));
  setInterval(async()=>{try{const data=await api("/api/rs232-test/status");render(data)}catch(_){}},1000);
});
