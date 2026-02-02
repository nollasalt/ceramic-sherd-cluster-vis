(function() {
  function ready(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  ready(function() {
    console.log('[modal] init start');

    function setupIfReady() {
      const modal = document.getElementById('image-modal');
      const modalImg = document.getElementById('modal-image');
      const closeBtn = document.getElementById('close-modal-btn');
      const zoomInBtn = document.getElementById('zoom-in-btn');
      const zoomOutBtn = document.getElementById('zoom-out-btn');
      const rotateLeftBtn = document.getElementById('rotate-left-btn');
      const rotateRightBtn = document.getElementById('rotate-right-btn');
      const resetBtn = document.getElementById('reset-btn');

      if (!modal || !modalImg) {
        console.warn('[modal] modal nodes missing, wait for layout render');
        return false;
      }

      let scale = 1;
      let rotation = 0;
      let offsetX = 0;
      let offsetY = 0;
      let isDragging = false;
      let startX = 0;
      let startY = 0;

      function applyTransform() {
        modalImg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale}) rotate(${rotation}deg)`;
      }

      function resetView() {
        scale = 1;
        rotation = 0;
        offsetX = 0;
        offsetY = 0;
        applyTransform();
      }

      function closeModal() {
        modal.style.display = 'none';
      }

      const basePath = (window._dash_config && window._dash_config.requests_pathname_prefix) || '/';

      async function loadAndShow(imgEl) {
        if (!imgEl) return;
        const imagePath = imgEl.dataset.imagePath || '';
        const inlineFull = imgEl.dataset.fullSrc;
        console.log('[modal] click', { imagePath, hasInline: !!inlineFull, src: imgEl.src });
        resetView();
        modal.style.display = 'block';
        modalImg.dataset.currentPath = imagePath;
        try {
          if (inlineFull) {
            console.log('[modal] using inline full src');
            modalImg.src = inlineFull;
            return;
          }
          if (imagePath) {
            const url = `${basePath.replace(/\/$/, '')}/get_full_image?image_path=${encodeURIComponent(imagePath)}`;
            console.log('[modal] fetching', url);
            const resp = await fetch(url);
            if (resp.ok) {
              const txt = await resp.text();
              console.log('[modal] fetch ok, length', txt.length);
              modalImg.src = txt;
              return;
            }
            console.warn('[modal] fetch failed', resp.status, resp.statusText);
          }
          modalImg.src = imgEl.src || '';
        } catch (err) {
          console.error('[modal] error loading', err);
          modalImg.src = imgEl.src || '';
        }
      }

      function bindImages() {
        const imgs = document.querySelectorAll('img[data-image-path]');
        let bound = 0;
        imgs.forEach((img) => {
          if (img.dataset.modalBound === '1') return;
          img.dataset.modalBound = '1';
          if (!img.style.cursor) img.style.cursor = 'pointer';
          img.addEventListener('click', () => loadAndShow(img));
          bound += 1;
        });
        if (bound > 0) console.log('[modal] bound images', bound);
      }

      bindImages();
      const observer = new MutationObserver(() => bindImages());
      observer.observe(document.body, { childList: true, subtree: true });

      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          closeModal();
        }
      });

      if (closeBtn) {
        closeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          closeModal();
        });
      }

      if (resetBtn) {
        resetBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          resetView();
        });
      }

      if (zoomInBtn) {
        zoomInBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          scale = Math.min(8, scale * 1.2);
          applyTransform();
        });
      }

      if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          scale = Math.max(0.2, scale / 1.2);
          applyTransform();
        });
      }

      if (rotateLeftBtn) {
        rotateLeftBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          rotation = (rotation - 90) % 360;
          applyTransform();
        });
      }

      if (rotateRightBtn) {
        rotateRightBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          rotation = (rotation + 90) % 360;
          applyTransform();
        });
      }

      modalImg.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 1.1 : 0.9;
        scale = Math.min(8, Math.max(0.2, scale * delta));
        applyTransform();
      }, { passive: false });

      modalImg.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDragging = true;
        startX = e.clientX - offsetX;
        startY = e.clientY - offsetY;
        modalImg.style.cursor = 'grabbing';
      });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        offsetX = e.clientX - startX;
        offsetY = e.clientY - startY;
        applyTransform();
      });

      window.addEventListener('mouseup', () => {
        isDragging = false;
        modalImg.style.cursor = 'move';
      });

      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          closeModal();
        }
      });

      console.log('[modal] init complete');
      return true;
    }

    if (setupIfReady()) return;

    console.log('[modal] wait for modal DOM...');
    const waitObserver = new MutationObserver(() => {
      if (setupIfReady()) {
        waitObserver.disconnect();
        console.log('[modal] DOM ready, observer stopped');
      }
    });
    waitObserver.observe(document.body || document.documentElement, { childList: true, subtree: true });
  });
})();
