(function () {
  if (window.__lineHoverHighlightInit) return;
  window.__lineHoverHighlightInit = true;

  function resolvePlotlyGraph(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    const gd = container.querySelector('.js-plotly-plot') || container;
    if (!gd || typeof gd.on !== 'function' || typeof Plotly === 'undefined') return null;
    return gd;
  }

  function setOpacity(elements, activeIndexSet, dimOpacity) {
    elements.forEach(function (el, index) {
      if (!el) return;
      el.style.opacity = activeIndexSet.has(index) ? '1' : String(dimOpacity);
    });
  }

  function clearOpacity(elements) {
    elements.forEach(function (el) {
      if (!el) return;
      el.style.opacity = '';
    });
  }

  function bindTrendChart(gd) {
    if (!gd || gd.dataset.hoverDimBound === '1') return;
    gd.dataset.hoverDimBound = '1';

    function getTraceElements() {
      return Array.from(gd.querySelectorAll('.scatterlayer .trace'));
    }

    function isHitAreaTrace(trace) {
      return trace && typeof trace.name === 'string' && trace.name.endsWith(' hit-area');
    }

    function getLinePaths(traceEl) {
      if (!traceEl) return [];
      return Array.from(traceEl.querySelectorAll('.lines path, path.js-line'));
    }

    function restore() {
      if (!gd.__trendDimActive) return;
      gd.__trendDimActive = false;
      gd.__trendActiveGroup = null;
      clearOpacity(getTraceElements());
    }

    function highlightByLegendGroup(group) {
      if (!group || !gd.data) return;
      if (gd.__trendDimActive && gd.__trendActiveGroup === group) return;
      const activeIndexes = new Set();
      gd.data.forEach(function (trace, index) {
        if (trace && trace.legendgroup === group) {
          activeIndexes.add(index);
        }
      });
      if (!activeIndexes.size) return;
      gd.__trendDimActive = true;
      gd.__trendActiveGroup = group;
      setOpacity(getTraceElements(), activeIndexes, 0.12);
    }

    function bindNativeLineHover() {
      const traceEls = getTraceElements();
      traceEls.forEach(function (traceEl, traceIndex) {
        const trace = gd.data && gd.data[traceIndex];
        if (!trace || !trace.legendgroup) return;

        const linePaths = getLinePaths(traceEl);
        if (!linePaths.length) return;

        linePaths.forEach(function (pathEl) {
          if (!pathEl) return;

          pathEl.style.pointerEvents = 'stroke';
          pathEl.style.cursor = 'pointer';

          if (isHitAreaTrace(trace)) {
            pathEl.style.strokeOpacity = '0.003';
          }

          if (pathEl.dataset.hoverBound === '1') return;
          pathEl.dataset.hoverBound = '1';

          pathEl.addEventListener('mouseenter', function () {
            highlightByLegendGroup(trace.legendgroup);
          });
        });
      });
    }

    gd.addEventListener('mousemove', function (event) {
      const target = event && event.target;
      if (!target || typeof target.closest !== 'function') {
        restore();
        return;
      }

      if (
        target.closest('.scatterlayer .trace .lines path') ||
        target.closest('.scatterlayer .trace path.js-line') ||
        target.closest('.hoverlayer')
      ) {
        return;
      }

      restore();
    });

    gd.addEventListener('mouseleave', function () {
      restore();
    });

    gd.on('plotly_afterplot', function () {
      bindNativeLineHover();
      restore();
    });

    bindNativeLineHover();
  }

  function bindSankey(gd) {
    if (!gd || gd.dataset.hoverDimBound === '1') return;
    gd.dataset.hoverDimBound = '1';

    function getLinkElements() {
      return Array.from(gd.querySelectorAll('.sankey-link'));
    }

    function restore() {
      if (!gd.__sankeyDimActive) return;
      gd.__sankeyDimActive = false;
      clearOpacity(getLinkElements());
    }

    function highlightLink(pointNumber) {
      const links = getLinkElements();
      if (!links.length) return;
      const activeIndexes = new Set([pointNumber]);
      gd.__sankeyDimActive = true;
      setOpacity(links, activeIndexes, 0.10);
    }

    gd.on('plotly_hover', function (eventData) {
      const point = eventData && eventData.points && eventData.points[0];
      const isLink = point &&
        Object.prototype.hasOwnProperty.call(point, 'source') &&
        Object.prototype.hasOwnProperty.call(point, 'target') &&
        typeof point.pointNumber === 'number';

      if (!isLink) {
        restore();
        return;
      }
      highlightLink(point.pointNumber);
    });

    gd.on('plotly_unhover', function () {
      restore();
    });

    gd.addEventListener('mousemove', function (event) {
      const target = event && event.target;
      if (!target || typeof target.closest !== 'function') {
        restore();
        return;
      }
      if (!target.closest('.sankey-link') && !target.closest('.hoverlayer')) {
        restore();
      }
    });

    gd.addEventListener('mouseleave', function () {
      restore();
    });

    gd.on('plotly_afterplot', function () {
      restore();
    });
  }

  function bindIfReady() {
    const trendChart = resolvePlotlyGraph('cluster-trend-chart');
    const sankeyChart = resolvePlotlyGraph('stratigraphy-sankey');

    if (trendChart) bindTrendChart(trendChart);
    if (sankeyChart) bindSankey(sankeyChart);
  }

  function init() {
    bindIfReady();
    const observer = new MutationObserver(bindIfReady);
    observer.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
