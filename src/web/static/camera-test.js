"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const message = (id, value) => { $(id).textContent = String(value ?? "").replace(/\*\*/g, "").replace(/`/g, ""); };
async function api(url, options={}) { const response=await fetch(url,{cache:"no-store",...options}); if(!response.ok){let error=`HTTP ${response.status}`;try{error=(await response.json()).detail||error}catch(_){} throw new Error(error)} return response.json(); }
const post=(url,data={})=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
const lightValues=()=>[1,2,3,4].map(i=>Number($(`light-${i}`).value));
const connection=()=>({port:$("light-port").value,baudrate:Number($("baudrate").value)});
let previewBusy=false, roiLive=false;

function previewImages(data) {
  if(data.image)$("cam-preview").src=data.image;else $("cam-preview").removeAttribute("src");
  if("roi_image" in data){
    if(data.roi_image)$("roi-preview").src=data.roi_image;else $("roi-preview").removeAttribute("src");
  }
  document.dispatchEvent(new CustomEvent("camera-preview-update",{detail:data}));
}

function renderReceipts(receipts, reset=false) {
  receipts?.forEach((note,index)=>{ if(reset || note != null) message(`light-receipt-${index+1}`,note || "최근 ACK: 미확인"); });
}
function renderRoi(data) {
  if(data.threshold != null){$("roi-threshold").value=data.threshold;$("roi-value").textContent=Number(data.threshold).toFixed(2)}
  $("roi-level").value=data.level||"custom";
  message("roi-details",data.details);
  if(data.message) message("roi-config-message",data.message);
}
function renderLightValues(values) {
  if(!values)return;
  values.forEach((value,index)=>{ $(`light-${index+1}`).value=value; $(`light-value-${index+1}`).textContent=`${value}%`; });
}
async function preview() {
  if(previewBusy)return;previewBusy=true;
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),5000);
  try{
    const includeRoi=roiLive || $("camera-viewer").dataset.source==="roi-preview";
    const data=await api(`/api/camera-test/preview?show_roi=${$("show-roi").checked}&include_roi=${includeRoi}`,{signal:controller.signal});
    previewImages(data);
    message("camera-status",data.camera_status);message("light-status",data.light_status);message("frame-info",data.frame_info);
  }catch(error){
    const info=error.name==="AbortError" ? "영상 응답 지연 · 연결을 다시 확인하고 있습니다." : `프리뷰 오류: ${error.message}`;
    message("frame-info",info);
    previewImages({image:null,roi_image:null,frame_info:info,roi_frame_info:info});
  }finally{clearTimeout(timeout);previewBusy=false}
}
async function lightApply(group=null) {
  try{const data=await post("/api/camera-test/light/apply",{...connection(),values:lightValues(),group});message("light-message",data.message);message("light-status",data.light_status);renderReceipts(data.receipts)}
  catch(error){message("light-message",`전송 실패: ${error.message}`)}
}
async function roiPreview(key,value) {
  try{renderRoi(await api(`/api/roi/preview?${key}=${encodeURIComponent(value)}`))}catch(error){message("roi-config-message",error.message)}
}

async function captureImages(messageId, filesId, folderButtonId) {
  const buttons=[$("preview-capture"),$("capture")];
  if(buttons.some(button=>button.disabled))return;
  buttons.forEach(button=>{button.disabled=true});
  message(messageId,"카메라 원본 프레임 저장 중…");
  $(filesId).replaceChildren();
  const folderButton=$(folderButtonId);
  folderButton.disabled=true;
  delete folderButton.dataset.folder;
  try{
    const data=await post("/api/camera-test/capture",{name:$("capture-name").value});
    message(messageId,data.message);
    $(filesId).innerHTML=data.files.map((file,index)=>`<a href="${esc(file.url)}" download>${index===0?"Color":"Mono"} PNG 다운로드</a>`).join("");
    if(data.folder && data.files.length){folderButton.dataset.folder=data.folder;folderButton.disabled=false}
  }catch(error){message(messageId,`이미지 저장 실패: ${error.message}`)}
  finally{buttons.forEach(button=>{button.disabled=false})}
}

async function openCaptureFolder(buttonId, messageId) {
  const button=$(buttonId);
  if(button.disabled || !button.dataset.folder)return;
  button.disabled=true;
  try{const data=await post("/api/camera-test/capture/open-folder",{folder:button.dataset.folder});message(messageId,data.message)}
  catch(error){message(messageId,`폴더 열기 실패: ${error.message}`)}
  finally{button.disabled=false}
}

document.addEventListener("DOMContentLoaded",async()=>{
  try{
    const data=await api("/api/camera-test/bootstrap");
    message("camera-status",data.camera_status);message("light-status",data.light_status);
    $("camera-serial").value=data.settings.serial_number;$("gain").value=data.settings.gain;$("fps").value=data.settings.fps;$("exposure").value=data.settings.exposure_us;
    $("light-port").value=data.light.port;$("baudrate").value=String(data.light.baudrate);
    $("port-options").innerHTML=data.ports.map(port=>`<option value="${esc(port)}">`).join("");
    $("light-groups").innerHTML=data.lighting.groups.map((group,index)=>`<div class="lighting-group-card"><div class="lighting-group-header"><span>LIGHT ${String(index+1).padStart(2,"0")}</span><strong>${esc(group.name)}</strong><code>${esc(group.channels)}</code></div><div class="brightness-value"><span>설정 밝기</span><strong id="light-value-${index+1}">0%</strong></div><input id="light-${index+1}" type="range" min="0" max="100" step="1"><button data-group="${index+1}">이 조명 적용</button><p id="light-receipt-${index+1}" class="hint">최근 ACK: 미확인</p></div>`).join("");
    renderLightValues(data.lighting.values);message("lighting-config-message",data.lighting.message);
    renderReceipts(data.light.receipts);
    [1,2,3,4].forEach(i=>{$(`light-${i}`).addEventListener("input",()=>{$(`light-value-${i}`).textContent=`${$(`light-${i}`).value}%`})});
    document.querySelectorAll("[data-group]").forEach(button=>button.addEventListener("click",()=>lightApply(Number(button.dataset.group))));
    renderRoi(await api("/api/roi"));
  }catch(error){message("camera-status",`시작 실패: ${error.message}`)}
  preview();setInterval(preview,200);
  $("preview-capture").addEventListener("click",()=>captureImages("preview-capture-message","preview-capture-files","preview-open-folder"));
  $("capture").addEventListener("click",()=>captureImages("capture-message","capture-files","capture-open-folder"));
  $("preview-open-folder").addEventListener("click",()=>openCaptureFolder("preview-open-folder","preview-capture-message"));
  $("capture-open-folder").addEventListener("click",()=>openCaptureFolder("capture-open-folder","capture-message"));
  $("apply-exposure").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/exposure",{exposure_us:Number($("exposure").value)});if(data.exposure_us!=null)$("exposure").value=data.exposure_us;message("camera-status",data.camera_status);message("exposure-message",data.message)}catch(error){message("exposure-message",error.message)}});
  $("load-exposure").addEventListener("click",async()=>{try{const data=await api("/api/camera-test/exposure");if(data.exposure_us!=null)$("exposure").value=data.exposure_us;message("exposure-message",data.message)}catch(error){message("exposure-message",error.message)}});
  $("extract-roi").addEventListener("click",async()=>{const button=$("extract-roi");button.disabled=true;try{const data=await post("/api/camera-test/roi");message("roi-message",data.message);roiLive=!!data.image;if(data.image)$("roi-preview").src=data.image;else $("roi-preview").removeAttribute("src")}catch(error){message("roi-message",error.message)}finally{button.disabled=false}});
  $("connect-camera").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/camera/connect",{serial_number:$("camera-serial").value,gain:Number($("gain").value),fps:Number($("fps").value)});message("camera-status",data.camera_status)}catch(error){message("camera-status",error.message)}});
  $("disconnect-camera").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/camera/disconnect");message("camera-status",data.camera_status);$("cam-preview").removeAttribute("src")}catch(error){message("camera-status",error.message)}});
  $("connect-light").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/light/connect",connection());message("light-message",data.message);message("light-status",data.light_status);if(data.message.includes("연결 완료"))renderReceipts([null,null,null,null],true)}catch(error){message("light-message",error.message)}});
  $("disconnect-light").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/light/disconnect");message("light-message",data.message);message("light-status",data.light_status);renderReceipts([null,null,null,null],true)}catch(error){message("light-message",error.message)}});
  $("apply-all").addEventListener("click",()=>lightApply());
  $("all-off").addEventListener("click",async()=>{try{const data=await post("/api/camera-test/light/off",connection());message("light-message",data.message);message("light-status",data.light_status);renderReceipts(data.receipts);data.reset_values.forEach((value,index)=>{if(value!=null){$(`light-${index+1}`).value=value;$(`light-value-${index+1}`).textContent=`${value}%`}})}catch(error){message("light-message",error.message)}});
  $("save-lighting").addEventListener("click",async()=>{try{const data=await post("/api/lighting/config",{values:lightValues()});message("lighting-config-message",data.message)}catch(error){message("lighting-config-message",error.message)}});
  $("load-lighting").addEventListener("click",async()=>{try{const data=await api("/api/lighting");renderLightValues(data.values);message("lighting-config-message",data.message)}catch(error){message("lighting-config-message",error.message)}});
  $("roi-level").addEventListener("change",()=>roiPreview("level",$("roi-level").value));
  $("roi-threshold").addEventListener("input",()=>{$("roi-value").textContent=$("roi-threshold").value;roiPreview("threshold",$("roi-threshold").value)});
  $("save-roi").addEventListener("click",async()=>{try{renderRoi(await post("/api/roi",{threshold:Number($("roi-threshold").value)}))}catch(error){message("roi-config-message",error.message)}});
  $("load-roi").addEventListener("click",async()=>{try{renderRoi(await api("/api/roi"))}catch(error){message("roi-config-message",error.message)}});
});
