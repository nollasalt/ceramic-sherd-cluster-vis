(function(){
  if (window.__inlineImageModalInit) return; window.__inlineImageModalInit = true;

  function ready(fn){ if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',fn);} else {fn();}}

  function init(){
    console.log('[modal-inline] init start');

    function setupIfReady(){
      const modal=document.getElementById('image-modal');
      const modalImg=document.getElementById('modal-image');
      const closeBtn=document.getElementById('close-modal-btn');
      const zoomInBtn=document.getElementById('zoom-in-btn');
      const zoomOutBtn=document.getElementById('zoom-out-btn');
      const rotateLeftBtn=document.getElementById('rotate-left-btn');
      const rotateRightBtn=document.getElementById('rotate-right-btn');
      const resetBtn=document.getElementById('reset-btn');
      if(!modal||!modalImg){ console.warn('[modal-inline] modal nodes missing, wait for layout render'); return false; }

      let scale=1, rotation=0, offsetX=0, offsetY=0, dragging=false, startX=0, startY=0;
      function apply(){ modalImg.style.transform=`translate(${offsetX}px, ${offsetY}px) scale(${scale}) rotate(${rotation}deg)`; }
      function reset(){ scale=1; rotation=0; offsetX=0; offsetY=0; apply(); }
      function close(){ modal.style.display='none'; }

      const basePath=(window._dash_config&&window._dash_config.requests_pathname_prefix)||'/';

      async function open(img){
        if(!img) return; const imagePath=img.dataset.imagePath||''; const inlineFull=img.dataset.fullSrc;
        console.log('[modal-inline] click', { imagePath, hasInline:!!inlineFull, src: img.src });
        reset(); modal.style.display='block'; modalImg.dataset.currentPath=imagePath;
        try{
          if(inlineFull){ console.log('[modal-inline] using inline full src'); modalImg.src=inlineFull; return; }
          if(imagePath){
            const url=`${basePath.replace(/\/$/,'')}/get_full_image?image_path=${encodeURIComponent(imagePath)}`;
            console.log('[modal-inline] fetching', url);
            const resp=await fetch(url);
            if(resp.ok){ const txt=await resp.text(); console.log('[modal-inline] fetch ok length', txt.length); modalImg.src=txt; return; }
            console.warn('[modal-inline] fetch failed', resp.status, resp.statusText);
          }
          modalImg.src=img.src||'';
        }catch(e){ console.error('[modal-inline] error loading', e); modalImg.src=img.src||''; }
      }

      function bindImgs(){
        let bound=0;
        document.querySelectorAll('img[data-image-path]').forEach((img)=>{
          if(img.dataset.modalBound==='1') return; img.dataset.modalBound='1'; if(!img.style.cursor) img.style.cursor='pointer';
          img.addEventListener('click',()=>open(img)); bound+=1;
        });
        if(bound>0) console.log('[modal-inline] bound images', bound);
      }

      bindImgs();
      const mo=new MutationObserver(bindImgs); mo.observe(document.body,{childList:true,subtree:true});

      modal.addEventListener('click',(e)=>{ if(e.target===modal) close(); });
      if(closeBtn) closeBtn.addEventListener('click',(e)=>{ e.stopPropagation(); close(); });
      if(resetBtn) resetBtn.addEventListener('click',(e)=>{ e.stopPropagation(); reset(); });
      if(zoomInBtn) zoomInBtn.addEventListener('click',(e)=>{ e.stopPropagation(); scale=Math.min(8,scale*1.2); apply(); });
      if(zoomOutBtn) zoomOutBtn.addEventListener('click',(e)=>{ e.stopPropagation(); scale=Math.max(0.2,scale/1.2); apply(); });
      if(rotateLeftBtn) rotateLeftBtn.addEventListener('click',(e)=>{ e.stopPropagation(); rotation=(rotation-90)%360; apply(); });
      if(rotateRightBtn) rotateRightBtn.addEventListener('click',(e)=>{ e.stopPropagation(); rotation=(rotation+90)%360; apply(); });

      modalImg.addEventListener('wheel',(e)=>{ e.preventDefault(); const d=e.deltaY<0?1.1:0.9; scale=Math.min(8,Math.max(0.2,scale*d)); apply(); },{passive:false});
      modalImg.addEventListener('mousedown',(e)=>{ e.preventDefault(); dragging=true; startX=e.clientX-offsetX; startY=e.clientY-offsetY; modalImg.style.cursor='grabbing'; });
      window.addEventListener('mousemove',(e)=>{ if(!dragging) return; offsetX=e.clientX-startX; offsetY=e.clientY-startY; apply(); });
      window.addEventListener('mouseup',()=>{ dragging=false; modalImg.style.cursor='move'; });
      window.addEventListener('keydown',(e)=>{ if(e.key==='Escape') close(); });

      console.log('[modal-inline] init complete');
      return true;
    }

    if (setupIfReady()) return;
    console.log('[modal-inline] wait for modal DOM...');
    const waitObserver = new MutationObserver(()=>{
      if(setupIfReady()){
        waitObserver.disconnect();
        console.log('[modal-inline] DOM ready, observer stopped');
      }
    });
    waitObserver.observe(document.body||document.documentElement,{childList:true,subtree:true});
  }

  ready(()=>{ try { init(); } catch(e) { console.error('modal init failed', e); } });
})();
