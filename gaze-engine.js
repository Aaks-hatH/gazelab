/*!
 * GazeEngine — a small, dependency-light wrapper around WebGazer.js that adds
 * the parts a real product needs on top of raw webcam gaze prediction:
 *
 *   - moving-average smoothing (raw WebGazer output is jittery, frame to frame)
 *   - outlier rejection (drops impossible jumps caused by blinks / tracking loss)
 *   - a confidence signal (fires when gaze samples stop arriving, e.g. face left frame)
 *   - dwell-time "stare to select" interactions — the standard input method used
 *     by real eye-gaze AAC (augmentative & alternative communication) devices
 *   - a plain event API (on/off/emit) so any UI — a game, a settings panel,
 *     a communication board — can consume gaze without touching WebGazer directly
 *
 * DROP-IN USAGE (works on any page, no build step):
 *
 *   <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
 *   <script src="gaze-engine.js"></script>
 *   <script>
 *     const gaze = new GazeEngine();
 *
 *     await gaze.start();                    // requests camera, boots tracking
 *
 *     gaze.on('gaze', ({x, y}) => {           // smoothed point, every frame
 *       cursorEl.style.left = x + 'px';
 *       cursorEl.style.top  = y + 'px';
 *     });
 *
 *     gaze.on('low-confidence', () => {       // face lost / tracking degraded
 *       showBanner('Center your face in the camera preview');
 *     });
 *
 *     // Calibration: place N dots on screen, have the user look + click each one
 *     // several times. WebGazer trains itself off real click coordinates —
 *     // registerClick() just re-emits a 'calibration-click' event for your UI.
 *     dotEl.addEventListener('click', (e) => gaze.registerClick(e.clientX, e.clientY));
 *
 *     // Dwell-to-select: stare at any element for `ms` to "click" it.
 *     gaze.dwellSelect(buttonEl, {
 *       ms: 900,
 *       onProgress: (pct) => ring.style.setProperty('--p', pct),
 *       onSelect: () => speak('Yes')
 *     });
 *
 *   </script>
 *
 * No React, no build tooling, no CSS required — this file only manages state
 * and math. All rendering stays in the host page.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.GazeEngine = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function GazeEngine(opts) {
    this.opts = Object.assign({
      smoothingWindow: 6,      // frames averaged together to steady the cursor
      outlierJumpPx: 420,      // ignore a jump bigger than this within outlierWindowMs
      outlierWindowMs: 120,
      lowConfidenceMs: 650,    // no samples for this long -> 'low-confidence'
      trackerType: 'TFFacemesh'
    }, opts || {});

    this._listeners = {};
    this._buffer = [];
    this._lastRaw = null;
    this._lastSampleTime = 0;
    this._dwellTargets = new Map();
    this._dwellRAF = null;
    this._confidenceTimer = null;
    this._started = false;
    this.current = null; // last smoothed {x,y}
  }

  var P = GazeEngine.prototype;

  /* ---------------- events ---------------- */

  P.on = function (event, cb) {
    (this._listeners[event] = this._listeners[event] || []).push(cb);
    return this;
  };

  P.off = function (event, cb) {
    if (!this._listeners[event]) return this;
    this._listeners[event] = this._listeners[event].filter(function (fn) { return fn !== cb; });
    return this;
  };

  P._emit = function (event, payload) {
    var list = this._listeners[event];
    if (!list) return;
    for (var i = 0; i < list.length; i++) {
      try { list[i](payload); } catch (e) { console.error('[GazeEngine] listener error:', e); }
    }
  };

  /* ---------------- lifecycle ---------------- */

  P.start = function () {
    var self = this;

    if (typeof webgazer === 'undefined') {
      return Promise.reject(new Error('WebGazer failed to load (check your connection and reload the page).'));
    }
    if (!window.isSecureContext) {
      return Promise.reject(new Error('Camera access requires HTTPS (or localhost). This page is not running in a secure context.'));
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return Promise.reject(new Error('This browser does not support camera access.'));
    }

    // Ask for the camera ourselves first, instead of letting WebGazer do it
    // internally. This gives us a real, specific error (permission denied, no
    // camera, camera already in use, etc.) instead of a silent hang, which is
    // the actual root cause of "allow was clicked but nothing happens": WebGazer's
    // own begin() can stall indefinitely if its tracking model fails to load,
    // with zero feedback to the page.
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 320 }, height: { ideal: 240 } }
    }).then(function (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      return self._bootWebgazer();
    });
  };

  P._bootWebgazer = function () {
    var self = this;

    webgazer.setRegression('ridge');
    if (webgazer.setTracker) { try { webgazer.setTracker(self.opts.trackerType); } catch (e) {} }
    if (webgazer.saveDataAcrossSessions) webgazer.saveDataAcrossSessions(false);

    var began = webgazer.begin();
    // Race begin() against a timeout: if the tracking model can't load (slow
    // network, blocked CDN, low-power mobile device) begin() otherwise never
    // resolves and the UI is stuck forever on "Requesting camera...".
    var timeout = new Promise(function (_, reject) {
      setTimeout(function () {
        reject(new Error('Camera timed out starting up. Check your connection and try again.'));
      }, 20000);
    });

    return Promise.race([began, timeout]).then(function () {
      webgazer.showVideoPreview(true);
      webgazer.showPredictionPoints(false);
      webgazer.showFaceOverlay(false);
      webgazer.showFaceFeedbackBox(false);
      webgazer.clearData();

      self._fixMobileVideo();

      webgazer.setGazeListener(function (data) {
        if (!data) return;
        self._ingest(data.x, data.y);
      });

      self._started = true;
      self._watchConfidence();
      self._runDwellLoop();
      self._emit('ready');
      return self;
    });
  };

  // iOS Safari (and some Android WebViews) refuse to actually play a <video>
  // element unless it is muted + playsinline. WebGazer's internal video feed
  // sets neither, so on mobile the camera light turns on (permission granted)
  // but the video frame never advances and no gaze data ever arrives.
  P._fixMobileVideo = function () {
    var vid = document.getElementById('webgazerVideoFeed');
    if (!vid) return;
    vid.setAttribute('playsinline', 'true');
    vid.setAttribute('webkit-playsinline', 'true');
    vid.muted = true;
    var p = vid.play();
    if (p && p.catch) p.catch(function () {});
  };

  P.pause = function () { if (typeof webgazer !== 'undefined') webgazer.pause(); };
  P.resume = function () { if (typeof webgazer !== 'undefined') webgazer.resume(); };

  P.destroy = function () {
    if (this._dwellRAF) cancelAnimationFrame(this._dwellRAF);
    if (this._confidenceTimer) clearInterval(this._confidenceTimer);
    this._dwellTargets.clear();
    this._listeners = {};
    if (typeof webgazer !== 'undefined') { try { webgazer.end(); } catch (e) {} }
    this._started = false;
  };

  P.isRunning = function () { return this._started; };

  /* ---------------- signal processing ---------------- */

  P._ingest = function (x, y) {
    var now = performance.now();

    if (this._lastRaw) {
      var dt = now - this._lastSampleTime;
      var dist = Math.hypot(x - this._lastRaw.x, y - this._lastRaw.y);
      if (dt < this.opts.outlierWindowMs && dist > this.opts.outlierJumpPx) {
        return; // discard: almost certainly a blink or momentary tracking glitch
      }
    }
    this._lastRaw = { x: x, y: y };
    this._lastSampleTime = now;

    this._buffer.push({ x: x, y: y });
    if (this._buffer.length > this.opts.smoothingWindow) this._buffer.shift();

    var sx = 0, sy = 0, n = this._buffer.length;
    for (var i = 0; i < n; i++) { sx += this._buffer[i].x; sy += this._buffer[i].y; }

    this.current = { x: sx / n, y: sy / n, raw: { x: x, y: y } };
    this._emit('gaze', this.current);
  };

  P._watchConfidence = function () {
    var self = this;
    clearInterval(this._confidenceTimer);
    this._confidenceTimer = setInterval(function () {
      if (!self._lastSampleTime) return;
      var idle = performance.now() - self._lastSampleTime;
      if (idle > self.opts.lowConfidenceMs) {
        self._emit('low-confidence', { idleMs: idle });
      }
    }, 300);
  };

  /* ---------------- calibration helpers ---------------- */

  // WebGazer trains itself from real click coordinates once begin() is active.
  // This just gives host UIs a clean event to hook progress bars, sounds, etc to.
  P.registerClick = function (x, y) {
    this._emit('calibration-click', { x: x, y: y });
  };

  P.clearCalibration = function () {
    if (typeof webgazer !== 'undefined') webgazer.clearData();
  };

  // Shows how accurate current calibration is: samples gaze for `sampleMs` while
  // the caller displays something at (xFrac, yFrac) of the viewport, and resolves
  // with the average pixel error.
  P.measureAccuracyAt = function (xFrac, yFrac, sampleMs) {
    var self = this;
    sampleMs = sampleMs || 800;
    return new Promise(function (resolve) {
      var targetX = xFrac * window.innerWidth;
      var targetY = yFrac * window.innerHeight;
      var samples = [];
      var collector = function (g) { samples.push(g); };
      self.on('gaze', collector);
      setTimeout(function () {
        self.off('gaze', collector);
        if (!samples.length) return resolve(null);
        var ax = 0, ay = 0;
        samples.forEach(function (s) { ax += s.x; ay += s.y; });
        ax /= samples.length; ay /= samples.length;
        resolve({
          distance: Math.hypot(ax - targetX, ay - targetY),
          point: { x: ax, y: ay },
          fpoint: { x: ax / window.innerWidth, y: ay / window.innerHeight }
        });
      }, sampleMs);
    });
  };

  /* ---------------- dwell-to-select ---------------- */
  // The core interaction for hands-free / speech-free control: look at something
  // long enough and it activates. This is how real eye-gaze AAC devices work.

  P.dwellSelect = function (el, config) {
    config = Object.assign({ ms: 900, onSelect: null, onProgress: null, onEnter: null, onLeave: null }, config);
    this._dwellTargets.set(el, Object.assign({ _acc: 0, _inside: false, _locked: false }, config));
    var self = this;
    return function () { self.cancelDwell(el); };
  };

  P.cancelDwell = function (el) {
    this._dwellTargets.delete(el);
  };

  P.setDwellTime = function (el, ms) {
    var cfg = this._dwellTargets.get(el);
    if (cfg) cfg.ms = ms;
  };

  P._runDwellLoop = function () {
    var self = this;
    var last = performance.now();

    function step() {
      var now = performance.now();
      var dt = now - last;
      last = now;

      if (self.current) {
        self._dwellTargets.forEach(function (cfg, el) {
          var r = el.getBoundingClientRect();
          var inside = self.current.x >= r.left && self.current.x <= r.right &&
                       self.current.y >= r.top && self.current.y <= r.bottom;

          if (inside) {
            if (!cfg._inside) {
              cfg._inside = true;
              cfg._acc = 0;
              cfg._locked = false;
              if (cfg.onEnter) cfg.onEnter();
            }
            if (!cfg._locked) {
              cfg._acc += dt;
              var pct = Math.min(1, cfg._acc / cfg.ms);
              if (cfg.onProgress) cfg.onProgress(pct);
              if (pct >= 1) {
                cfg._locked = true;
                if (cfg.onSelect) cfg.onSelect();
              }
            }
          } else if (cfg._inside) {
            cfg._inside = false;
            cfg._acc = 0;
            cfg._locked = false;
            if (cfg.onLeave) cfg.onLeave();
            if (cfg.onProgress) cfg.onProgress(0);
          }
        });
      }
      self._dwellRAF = requestAnimationFrame(step);
    }
    this._dwellRAF = requestAnimationFrame(step);
  };

  return GazeEngine;
}));
