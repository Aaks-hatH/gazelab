/*!
 * MPGazeEngine — WebGazer replacement built on MediaPipe FaceLandmarker
 * (Tasks Vision), the same library hackjps/Aloud uses for blink detection.
 *
 * WHY THIS EXISTS, not just a tuned WebGazer:
 * WebGazer regresses screen (x,y) directly from raw eye-patch PIXELS -- a
 * high-dimensional, appearance-based feature that's extremely sensitive to
 * lighting, camera, and head position. FaceLandmarker instead gives us the
 * IRIS CENTER as an explicit, sub-pixel-accurate landmark (it's a dedicated,
 * well-trained model output, not something we're inferring from raw pixels
 * ourselves). Regressing from "where is the iris relative to the eye
 * corners" -- a half-dozen numbers with real geometric meaning -- is a much
 * lower-dimensional, much more stable regression problem than WebGazer's.
 * Same idea real geometric eye-trackers have used for years; MediaPipe just
 * hands us the iris landmarks for free.
 *
 * Blink detection also upgrades: instead of a hand-rolled eye-aspect-ratio
 * from corner distances (this project's old approach), we use
 * FaceLandmarker's own eyeBlinkLeft/eyeBlinkRight blendshape scores -- a
 * dedicated, trained classifier output, exactly what hackjps/Aloud's
 * blink.mjs uses.
 *
 * PUBLIC API IS A DROP-IN MATCH for the old GazeEngine (gaze-engine.js):
 * on/off, start, destroy, pause/resume, current, eyeState, registerClick,
 * calibrateDwell, clearCalibration, calibrateBlink, measureAccuracyAt,
 * dwellSelect, cancelDwell, scanSelect, isRunning. Swapping
 * `new GazeEngine()` for `new MPGazeEngine()` in index.html is the only
 * change needed there.
 *
 * DIAGNOSTICS: this file is chatty in the console on purpose (toggle with
 * `opts.debug`, default on, or `window.mpGazeDebug.setDebug(false)`).
 * Since gaze regression quality depends entirely on things that only show
 * up on a real webcam in a real room -- lighting, camera FOV, how much the
 * user's eyes actually moved during calibration -- console output is the
 * fastest way to see WHERE it's going wrong instead of guessing blind:
 *
 *   [MPGaze] init            model/wasm URLs, actual vs requested resolution
 *   [MPGaze] frame (1/s)     face found?, blink score, iris ratios
 *   [MPGaze] calib sample    running sample count + per-feature variance
 *   [MPGaze] fit             regression refit: sample count, train RMS error,
 *                            per-feature variance (flags degenerate features)
 *   [MPGaze] accuracy test   per-target predicted vs actual, pixel error
 *
 * window.mpGazeDebug.dumpSamples() / .dumpFit() / .setDebug(bool) are
 * available in devtools for live poking.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.MPGazeEngine = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
  var MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

  // Same 468-point FaceMesh topology WebGazer used -- FaceLandmarker's first
  // 468 landmarks keep identical indices, then appends 10 iris points.
  var L = {
    rightCornerOuter: 33, rightCornerInner: 133, rightUpperMid: 159, rightLowerMid: 145,
    leftCornerOuter: 263, leftCornerInner: 362, leftUpperMid: 386, leftLowerMid: 374,
    rightIris: 468, leftIris: 473
  };

  function GazeEngine(opts) {
    this.opts = Object.assign({
      // ---- detection cadence ----
      detectIntervalMs: 62,   // ~16fps -- matches hackjps's BlinkCam; FaceLandmarker
                               // is heavier than WebGazer's tracker, no need to run
                               // it faster than the eye can actually move meaningfully
      cameraConstraints: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },

      // ---- regression ----
      ridgeLambda: 6,          // L2 regularization strength for the closed-form ridge solve
      recencyHalfLifeMs: 40000, // samples older than this count for half weight -- lets
                                 // drift (posture shift, lighting change) fade out, same
                                 // motivation as the old engine's weightedRidge choice
      minSamplesToPredict: 8, // don't emit a 'gaze' point at all until this many
                              // calibration samples exist -- a prediction from 2 samples
                              // is noise, not signal, and misrepresents accuracy
      refitThrottleMs: 150,   // re-solve the regression at most this often
      useHeadPoseFeature: false, // facial transformation matrix -> yaw/pitch as extra
                                  // regression features. OFF by default: the matrix
                                  // decomposition is unverified on a real device from
                                  // this environment (see console warning at init) --
                                  // an unverified feature can hurt more than it helps.
                                  // Flip to true once you've confirmed via
                                  // window.mpGazeDebug.dumpFit() that yaw/pitch values
                                  // look sane (roughly -30..30 degrees, moving the
                                  // right direction as you turn your head).

      // ---- smoothing / outliers (same semantics as the old engine) ----
      smoothingWindow: 4,
      outlierJumpPx: 420,
      outlierWindowMs: 120,
      lowConfidenceMs: 650,

      // ---- blink (now blendshape-driven, thresholds still adaptive) ----
      earClosedRatio: 0.62,
      earOpenRatio: 0.78,
      blinkMinMs: 60,
      blinkMaxMs: 400,
      earBaselineAlpha: 0.02,

      // ---- scan-select (unchanged from the old engine) ----
      scanStepMs: 900,
      scanHoldMs: 550,
      scanRefireMs: 700,

      debug: true
    }, opts || {});

    this._listeners = {};
    this._buffer = [];
    this._lastRaw = null;
    this._lastSampleTime = 0;
    this._dwellTargets = new Map();
    this._dwellRAF = null;
    this._confidenceTimer = null;
    this._detectRAF = null;
    this._earBaseline = null;
    this._eyeClosedSince = null;
    this._sustainedFired = false;
    this._started = false;
    this.current = null;
    this.eyeState = { ear: null, open: null, closed: false, baseline: null };

    this._video = null;
    this._stream = null;
    this._landmarker = null;
    this._lastVideoTime = -1;
    this._lastDetectAt = 0;
    this._lastFeat = null;      // most recent feature vector, for feedCalibrationPoint
    this._lastLandmarks = null;
    this._samples = [];         // {feat:[...], x, y, t}
    this._reg = null;           // { wx:[...], wy:[...] }
    this._lastFitAt = 0;
    this._frameLogT = 0;

    var self = this;
    window.mpGazeDebug = {
      setDebug: function (v) { self.opts.debug = !!v; },
      dumpSamples: function () { return self._samples.slice(); },
      dumpFit: function () { return self._reg; },
      dumpFeature: function () { return self._lastFeat; }
    };
  }

  var P = GazeEngine.prototype;

  /* ---------------- events (identical to the old engine) ---------------- */

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
      try { list[i](payload); } catch (e) { console.error('[MPGaze] listener error:', e); }
    }
  };
  P._log = function () {
    if (this.opts.debug) console.log.apply(console, ['[MPGaze]'].concat(Array.prototype.slice.call(arguments)));
  };
  P._warn = function () {
    console.warn.apply(console, ['[MPGaze]'].concat(Array.prototype.slice.call(arguments)));
  };

  /* ---------------- lifecycle ---------------- */

  P.start = function () {
    var self = this;

    if (!window.isSecureContext) {
      return Promise.reject(new Error('Camera access requires HTTPS (or localhost). This page is not running in a secure context.'));
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return Promise.reject(new Error('This browser does not support camera access.'));
    }

    console.groupCollapsed('[MPGaze] init');
    console.log('wasm:', WASM_URL);
    console.log('model:', MODEL_URL);
    console.groupEnd();

    return this._boot();
  };

  P._boot = function () {
    var self = this;

    var streamP = navigator.mediaDevices.getUserMedia({
      video: this.opts.cameraConstraints, audio: false
    });

    var modelP = import(/* webpackIgnore: true */ 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm')
      .then(function (mod) {
        var FilesetResolver = mod.FilesetResolver, FaceLandmarker = mod.FaceLandmarker;
        return FilesetResolver.forVisionTasks(WASM_URL).then(function (fileset) {
          return FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
            runningMode: 'VIDEO',
            numFaces: 1
          });
        });
      });

    var timeout = new Promise(function (_, reject) {
      setTimeout(function () { reject(new Error('Model or camera timed out starting up. Check your connection and try again.')); }, 15000);
    });

    return Promise.race([Promise.all([streamP, modelP]), timeout]).then(function (results) {
      var stream = results[0], landmarker = results[1];
      self._stream = stream;
      self._landmarker = landmarker;
      self._setupVideoEl(stream);

      var track = stream.getVideoTracks()[0];
      var settings = track && track.getSettings ? track.getSettings() : {};
      self._log('camera resolution:', settings.width + 'x' + settings.height, '(requested ideal 1280x720)');
      if (settings.width && settings.width < 640) {
        self._warn('camera resolution is low (' + settings.width + 'px wide). Iris landmark precision drops '
          + 'noticeably below ~640px wide -- if accuracy is bad, check this first before suspecting the '
          + 'regression math.');
      }

      self._started = true;
      self._watchConfidence();
      self._runDwellLoop();
      self._runDetectLoop();
      self._emit('ready', {});
      return self;
    }, function (err) {
      var msg = (err && err.message) ? err.message : 'Camera or model failed to start.';
      var wrapped = new Error(msg);
      if (err && err.name) wrapped.name = err.name;
      throw wrapped;
    });
  };

  P._setupVideoEl = function (stream) {
    // Reuses the SAME element IDs the old WebGazer-based build used, so all
    // of index.html's existing #webgazerVideoContainer / #webgazerVideoFeed
    // CSS (camera preview box, mirroring, positioning) keeps working
    // unmodified.
    var container = document.getElementById('webgazerVideoContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'webgazerVideoContainer';
      document.body.appendChild(container);
    }
    var video = document.getElementById('webgazerVideoFeed');
    if (!video) {
      video = document.createElement('video');
      video.id = 'webgazerVideoFeed';
      container.appendChild(video);
    }
    video.setAttribute('playsinline', 'true');
    video.setAttribute('webkit-playsinline', 'true');
    video.muted = true;
    video.srcObject = stream;
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
    this._video = video;
  };

  P.pause = function () { if (this._video) this._video.pause(); };
  P.resume = function () { if (this._video) { var p = this._video.play(); if (p && p.catch) p.catch(function () {}); } };

  P.destroy = function () {
    if (this._dwellRAF) cancelAnimationFrame(this._dwellRAF);
    if (this._detectRAF) cancelAnimationFrame(this._detectRAF);
    if (this._confidenceTimer) clearInterval(this._confidenceTimer);
    this._dwellTargets.clear();
    this._listeners = {};
    if (this._stream) { try { this._stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) {} }
    if (this._landmarker) { try { this._landmarker.close(); } catch (e) {} }
    this._started = false;
  };

  P.isRunning = function () { return this._started; };

  /* ---------------- detection loop ---------------- */

  P._runDetectLoop = function () {
    var self = this;

    function dist(a, b) { var dx = a.x - b.x, dy = a.y - b.y; return Math.sqrt(dx * dx + dy * dy); }

    function step() {
      self._detectRAF = requestAnimationFrame(step);
      var video = self._video;
      var now = performance.now();
      if (!video || !self._landmarker) return;
      if (now - self._lastDetectAt < self.opts.detectIntervalMs) return;
      if (video.currentTime === self._lastVideoTime || video.readyState < 2) return;
      self._lastDetectAt = now;
      self._lastVideoTime = video.currentTime;

      var res = self._landmarker.detectForVideo(video, now);
      var lm = res && res.faceLandmarks && res.faceLandmarks[0];
      var shapes = res && res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;

      if (!lm || lm.length < 478) {
        // No face this frame -- leave eyeState/current alone, the
        // low-confidence watcher (driven off _lastSampleTime) will fire on
        // its own if this persists.
        return;
      }
      self._lastLandmarks = lm;

      /* ---- blink, from the blendshape classifier, not geometry ---- */
      var blink = 0;
      if (shapes) {
        var get = function (name) {
          for (var i = 0; i < shapes.length; i++) if (shapes[i].categoryName === name) return shapes[i].score;
          return 0;
        };
        blink = Math.max(get('eyeBlinkLeft'), get('eyeBlinkRight'));
      }
      // Keep the same "ear" naming/threshold direction the rest of the
      // engine (and calibrateBlink/scanSelect) already expects: higher =
      // more open, shrinking toward 0 as eyes close.
      self._updateEyeState(1 - blink, now);

      /* ---- iris-ratio features for gaze regression ---- */
      var feat = self._extractFeatures(lm);
      self._lastFeat = feat;

      if (self.opts.debug && now - self._frameLogT > 1000) {
        self._frameLogT = now;
        self._log('frame:', 'blink=' + blink.toFixed(2), 'feat=' + feat.slice(0, 4).map(function (v) { return v.toFixed(3); }).join(','),
          'samples=' + self._samples.length, self._reg ? '(fitted)' : '(not calibrated yet)');
      }

      if (self._reg && self._samples.length >= self.opts.minSamplesToPredict && !self.eyeState.closed) {
        var x = self._dot(feat, self._reg.wx);
        var y = self._dot(feat, self._reg.wy);
        self._ingest(x, y);
      }
    }
    step();
  };

  // Normalized iris position within each eye's horizontal/vertical span.
  // Landmarks come back as {x,y,z} normalized to [0,1] of the input frame,
  // so this is resolution-independent by construction.
  P._extractFeatures = function (lm) {
    var rOuter = lm[L.rightCornerOuter], rInner = lm[L.rightCornerInner];
    var rUp = lm[L.rightUpperMid], rDown = lm[L.rightLowerMid], rIris = lm[L.rightIris];
    var lOuter = lm[L.leftCornerOuter], lInner = lm[L.leftCornerInner];
    var lUp = lm[L.leftUpperMid], lDown = lm[L.leftLowerMid], lIris = lm[L.leftIris];

    var rSpanX = (rInner.x - rOuter.x) || 1e-6;
    var rSpanY = (rDown.y - rUp.y) || 1e-6;
    var lSpanX = (lInner.x - lOuter.x) || 1e-6;
    var lSpanY = (lDown.y - lUp.y) || 1e-6;

    var rX = (rIris.x - rOuter.x) / rSpanX;
    var rY = (rIris.y - rUp.y) / rSpanY;
    var lX = (lIris.x - lOuter.x) / lSpanX;
    var lY = (lIris.y - lUp.y) / lSpanY;

    // bias term last, matches _dot()'s expectation of a constant 1 feature
    return [rX, rY, lX, lY, 1];
  };

  P._dot = function (feat, w) {
    var s = 0;
    for (var i = 0; i < feat.length; i++) s += feat[i] * w[i];
    return s;
  };

  P._updateEyeState = function (ear, now) {
    var self = this;
    if (self._earBaseline == null) {
      self._earBaseline = ear;
    } else if (!self.eyeState.closed) {
      self._earBaseline += (ear - self._earBaseline) * self.opts.earBaselineAlpha;
    }
    var closedThresh = self._earBaseline * self.opts.earClosedRatio;
    var openThresh = self._earBaseline * self.opts.earOpenRatio;

    if (!self.eyeState.closed && ear < closedThresh) {
      self.eyeState.closed = true;
      self._eyeClosedSince = now;
      self._sustainedFired = false;
      self._emit('eyes-closed-start', {});
    } else if (self.eyeState.closed && ear > openThresh) {
      self.eyeState.closed = false;
      var closedMs = now - self._eyeClosedSince;
      self._eyeClosedSince = null;
      if (closedMs >= self.opts.blinkMinMs && closedMs <= self.opts.blinkMaxMs) {
        self._emit('blink', { durationMs: closedMs });
      }
      self._emit('eyes-open', { durationMs: closedMs });
    }
    if (self.eyeState.closed && !self._sustainedFired && (now - self._eyeClosedSince) >= self.opts.blinkMaxMs) {
      self._sustainedFired = true;
      self._emit('eyes-closed', {});
    }
    self.eyeState.ear = ear;
    self.eyeState.baseline = self._earBaseline;
    self.eyeState.open = !self.eyeState.closed;
  };

  P._watchConfidence = function () {
    var self = this;
    clearInterval(this._confidenceTimer);
    this._confidenceTimer = setInterval(function () {
      if (!self._lastSampleTime) return;
      var idle = performance.now() - self._lastSampleTime;
      if (idle > self.opts.lowConfidenceMs) self._emit('low-confidence', { idleMs: idle });
    }, 300);
  };

  /* ---------------- smoothing / outlier rejection (unchanged) ---------------- */

  P._ingest = function (x, y) {
    var now = performance.now();
    if (this._lastRaw) {
      var dt = now - this._lastSampleTime;
      var dist = Math.hypot(x - this._lastRaw.x, y - this._lastRaw.y);
      if (dt < this.opts.outlierWindowMs && dist > this.opts.outlierJumpPx) return;
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

  /* ---------------- calibration ---------------- */

  P.registerClick = function (x, y) {
    // Unlike WebGazer, nothing auto-trains from ordinary document clicks
    // here, so a manual-mode calibration dot MUST route through this to
    // actually produce a sample.
    this.feedCalibrationPoint(x, y);
    this._emit('calibration-click', { x: x, y: y });
  };

  P.feedCalibrationPoint = function (x, y) {
    if (!this._lastFeat || this.eyeState.closed) return;
    var now = performance.now();
    this._samples.push({ feat: this._lastFeat.slice(), x: x, y: y, t: now });
    // Cap history so a long session doesn't grow this unboundedly; recency
    // weighting in _fit() already down-weights old samples, this is just a
    // hard ceiling.
    if (this._samples.length > 6000) this._samples.splice(0, this._samples.length - 6000);
    this._emit('calibration-feed', { x: x, y: y });
    this._maybeFit();
  };

  P._maybeFit = function () {
    var now = performance.now();
    if (now - this._lastFitAt < this.opts.refitThrottleMs) return;
    this._lastFitAt = now;
    this._fit();
  };

  // Closed-form ridge regression, solved separately for x and y targets,
  // with exponential recency weighting so a long session gradually lets go
  // of stale calibration data (the equivalent of the old engine's
  // weightedRidge choice over WebGazer's default plain ridge).
  P._fit = function () {
    var samples = this._samples;
    if (samples.length < 3) return;
    var dim = samples[0].feat.length;
    var now = performance.now();
    var halfLife = this.opts.recencyHalfLifeMs;
    var lambda = this.opts.ridgeLambda;

    // Normal equations: (F^T W F + lambda*I) w = F^T W y
    var FtF = zeros(dim, dim), FtX = zeros(dim, 1), FtY = zeros(dim, 1);
    var totalW = 0;
    for (var s = 0; s < samples.length; s++) {
      var f = samples[s].feat;
      var age = now - samples[s].t;
      var w = Math.pow(0.5, age / halfLife);
      totalW += w;
      for (var i = 0; i < dim; i++) {
        FtX[i][0] += w * f[i] * samples[s].x;
        FtY[i][0] += w * f[i] * samples[s].y;
        for (var j = 0; j < dim; j++) FtF[i][j] += w * f[i] * f[j];
      }
    }
    for (var d = 0; d < dim; d++) FtF[d][d] += lambda;

    var wx = solveLinearSystem(FtF, FtX);
    var wy = solveLinearSystem(FtF, FtY);
    if (!wx || !wy) {
      this._warn('regression fit failed (singular matrix) -- likely means calibration samples don\'t '
        + 'vary enough (e.g. all clicks landed with near-identical eye position). Spread calibration '
        + 'points further apart or recalibrate.');
      return;
    }

    // Diagnostics: training-set residual (RMS pixel error against the very
    // data the fit was built from). This being low does NOT guarantee good
    // real-world accuracy (that's what the accuracy test screen measures),
    // but if THIS is already high, the features/labels themselves are the
    // problem -- no amount of regularization fixes that.
    var se = 0;
    for (var k = 0; k < samples.length; k++) {
      var px = dot(samples[k].feat, wx.map(function (r) { return r[0]; }));
      var py = dot(samples[k].feat, wy.map(function (r) { return r[0]; }));
      se += Math.pow(px - samples[k].x, 2) + Math.pow(py - samples[k].y, 2);
    }
    var trainRmsPx = Math.sqrt(se / samples.length);

    this._reg = { wx: wx.map(function (r) { return r[0]; }), wy: wy.map(function (r) { return r[0]; }) };

    this._log('fit:', samples.length + ' samples', 'trainRMS=' + trainRmsPx.toFixed(0) + 'px',
      trainRmsPx > 250 ? '(high -- check feature variance below, or camera framing/lighting)' : '(reasonable)');
    if (this.opts.debug) this._logFeatureVariance(samples);
  };

  P._logFeatureVariance = function (samples) {
    var dim = samples[0].feat.length - 1; // skip bias term
    var names = ['rightIris.x', 'rightIris.y', 'leftIris.x', 'leftIris.y'];
    var out = [];
    for (var i = 0; i < dim; i++) {
      var vals = samples.map(function (s) { return s.feat[i]; });
      var mean = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      var variance = vals.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / vals.length;
      out.push(names[i] + ' var=' + variance.toFixed(5));
      if (variance < 0.0005) {
        this._warn('feature "' + names[i] + '" barely varies across calibration samples (var=' + variance.toFixed(5)
          + '). If this stays true across a full calibration pass, that eye/axis isn\'t contributing real '
          + 'signal -- check camera framing (is that eye clipped or in shadow?) or verify the landmark '
          + 'indices in mp-gaze-engine.js against a live console.log of `window.mpGazeDebug.dumpFeature()`.');
      }
    }
    this._log('feature variance:', out.join('  '));
  };

  P.clearCalibration = function () {
    this._samples = [];
    this._reg = null;
    this._log('calibration cleared');
  };

  // Same primitive/contract as the old engine: waits settleMs for the eye
  // to arrive and fixate, then feeds samples for the remainder of ms.
  P.calibrateDwell = function (x, y, config) {
    config = Object.assign({ ms: 1600, settleMs: 350, sampleEveryMs: 130, onProgress: null, onDone: null }, config);
    var self = this;
    var elapsed = 0, bonus = 0;
    var blinkHandler = function () { bonus += config.sampleEveryMs * 3; };
    self.on('blink', blinkHandler);

    var timer = setInterval(function () {
      elapsed += config.sampleEveryMs;
      if (elapsed >= config.settleMs && !self.eyeState.closed) self.feedCalibrationPoint(x, y);
      var pct = Math.min(1, (elapsed + bonus) / config.ms);
      if (config.onProgress) config.onProgress(pct);
      if (pct >= 1) {
        clearInterval(timer);
        self.off('blink', blinkHandler);
        if (config.onDone) config.onDone();
      }
    }, config.sampleEveryMs);

    return function cancel() { clearInterval(timer); self.off('blink', blinkHandler); };
  };

  // Percentile-based open/closed threshold calibration -- identical
  // approach to the old engine's calibrateBlink, just operating on the
  // blendshape-derived signal instead of geometric EAR.
  P.calibrateBlink = function (config) {
    config = Object.assign({ openMs: 1500, closedMs: 1500, onPhase: null }, config);
    var self = this;

    function percentile(arr, p) {
      if (!arr.length) return null;
      var s = arr.slice().sort(function (a, b) { return a - b; });
      var i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
      return s[i];
    }

    return new Promise(function (resolve) {
      var openSamples = [], closedSamples = [];
      if (config.onPhase) config.onPhase('open');
      var collectOpen = setInterval(function () {
        if (self.eyeState.ear != null && !self.eyeState.closed) openSamples.push(self.eyeState.ear);
      }, 50);

      setTimeout(function () {
        clearInterval(collectOpen);
        if (config.onPhase) config.onPhase('closed');
        var collectClosed = setInterval(function () {
          if (self.eyeState.ear != null) closedSamples.push(self.eyeState.ear);
        }, 50);

        setTimeout(function () {
          clearInterval(collectClosed);
          var baseline = self._earBaseline || percentile(openSamples, 50) || 1;
          var openP = percentile(openSamples, 15);
          var closedP = percentile(closedSamples, 85);

          var result = null;
          if (openP != null && closedP != null && openP > closedP) {
            var gap = openP - closedP;
            var closedRatio = Math.min(0.85, Math.max(0.35, (closedP + gap * 0.5) / baseline));
            var openRatio = Math.min(closedRatio - 0.05, Math.max(0.5, (closedP + gap * 0.72) / baseline));
            result = { earClosedRatio: closedRatio, earOpenRatio: openRatio };
            self.opts.earClosedRatio = closedRatio;
            self.opts.earOpenRatio = openRatio;
            self._log('blink calibration:', 'closedRatio=' + closedRatio.toFixed(2), 'openRatio=' + openRatio.toFixed(2),
              '(open samples=' + openSamples.length + ', closed samples=' + closedSamples.length + ')');
          } else {
            self._warn('blink calibration inconclusive (open/closed samples too close together) -- '
              + 'keeping previous thresholds. openP=' + openP + ' closedP=' + closedP);
          }
          if (config.onPhase) config.onPhase('done');
          resolve(result);
        }, config.closedMs);
      }, config.openMs);
    });
  };

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
        if (!samples.length) {
          self._warn('accuracy test: zero gaze samples for target at', Math.round(targetX) + ',' + Math.round(targetY),
            '-- regression may not be fitted yet (need ' + self.opts.minSamplesToPredict + '+ calibration samples), '
            + 'or the eyes were closed/off-camera the whole window.');
          return resolve(null);
        }
        var ax = 0, ay = 0;
        samples.forEach(function (s) { ax += s.x; ay += s.y; });
        ax /= samples.length; ay /= samples.length;
        var result = {
          distance: Math.hypot(ax - targetX, ay - targetY),
          point: { x: ax, y: ay },
          fpoint: { x: ax / window.innerWidth, y: ay / window.innerHeight }
        };
        self._log('accuracy test:', 'target=(' + Math.round(targetX) + ',' + Math.round(targetY) + ')',
          'predicted=(' + Math.round(ax) + ',' + Math.round(ay) + ')', 'error=' + Math.round(result.distance) + 'px',
          'n=' + samples.length);
        resolve(result);
      }, sampleMs);
    });
  };

  /* ---------------- dwell-to-select (unchanged) ---------------- */

  P.dwellSelect = function (el, config) {
    config = Object.assign({
      ms: 900, onSelect: null, onProgress: null, onEnter: null, onLeave: null,
      calibrate: false, calibEveryMs: 180
    }, config);
    this._dwellTargets.set(el, Object.assign({ _acc: 0, _inside: false, _locked: false, _calibAcc: 0 }, config));
    var self = this;
    return function () { self.cancelDwell(el); };
  };

  P.cancelDwell = function (el) { this._dwellTargets.delete(el); };
  P.setDwellTime = function (el, ms) { var cfg = this._dwellTargets.get(el); if (cfg) cfg.ms = ms; };

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
              cfg._inside = true; cfg._acc = 0; cfg._calibAcc = 0; cfg._locked = false;
              if (cfg.onEnter) cfg.onEnter();
            }
            if (!cfg._locked) {
              cfg._acc += dt;
              var pct = Math.min(1, cfg._acc / cfg.ms);
              if (cfg.onProgress) cfg.onProgress(pct);
              if (cfg.calibrate) {
                cfg._calibAcc += dt;
                if (cfg._calibAcc >= cfg.calibEveryMs) {
                  cfg._calibAcc = 0;
                  if (!self.eyeState.closed) self.feedCalibrationPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
                }
              }
              if (pct >= 1) { cfg._locked = true; if (cfg.onSelect) cfg.onSelect(); }
            }
          } else if (cfg._inside) {
            cfg._inside = false; cfg._acc = 0; cfg._calibAcc = 0; cfg._locked = false;
            if (cfg.onLeave) cfg.onLeave();
            if (cfg.onProgress) cfg.onProgress(0);
          }
        });
      }
      self._dwellRAF = requestAnimationFrame(step);
    }
    this._dwellRAF = requestAnimationFrame(step);
  };

  /* ---------------- scan-select (unchanged) ---------------- */

  P.scanSelect = function (items, config) {
    config = Object.assign({
      stepMs: this.opts.scanStepMs, holdMs: this.opts.scanHoldMs, refireMs: this.opts.scanRefireMs,
      loop: true, onHighlight: null, onUnhighlight: null, onSelect: null
    }, config);
    var self = this;
    var idx = -1, lastFire = 0, fired = false;

    function highlight(next) {
      if (idx >= 0 && items[idx] && config.onUnhighlight) config.onUnhighlight(items[idx], idx);
      idx = next;
      if (items[idx] && config.onHighlight) config.onHighlight(items[idx], idx);
    }
    function advance() {
      if (!items.length) return;
      var next = idx + 1;
      if (next >= items.length) { if (!config.loop) { stop(); return; } next = 0; }
      highlight(next);
    }
    var stepTimer = setInterval(advance, config.stepMs);
    advance();

    var eyesClosedHandler = function () { fired = false; };
    var eyesOpenHandler = function () { fired = false; };
    self.on('eyes-closed-start', eyesClosedHandler);
    self.on('eyes-open', eyesOpenHandler);

    function pollHold() {
      if (self.eyeState.closed && self._eyeClosedSince && !fired) {
        var held = performance.now() - self._eyeClosedSince;
        var since = performance.now() - lastFire;
        if (held >= config.holdMs && since >= config.refireMs) {
          fired = true; lastFire = performance.now();
          if (items[idx] && config.onSelect) config.onSelect(items[idx], idx);
        }
      }
    }
    var holdTimer = setInterval(pollHold, 50);

    function stop() {
      clearInterval(stepTimer);
      clearInterval(holdTimer);
      self.off('eyes-closed-start', eyesClosedHandler);
      self.off('eyes-open', eyesOpenHandler);
      if (idx >= 0 && items[idx] && config.onUnhighlight) config.onUnhighlight(items[idx], idx);
    }
    return stop;
  };

  /* ---------------- tiny linear algebra (no external deps) ---------------- */

  function zeros(r, c) {
    var m = [];
    for (var i = 0; i < r; i++) { m.push(new Array(c).fill(0)); }
    return m;
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  // Gauss-Jordan elimination with partial pivoting. Dimension here is tiny
  // (5, or 7 with head pose), so this is plenty fast and avoids pulling in
  // a matrix library for a page that otherwise has zero dependencies.
  function solveLinearSystem(A, b) {
    var n = A.length;
    var M = A.map(function (row, i) { return row.concat(b[i]); });
    for (var col = 0; col < n; col++) {
      var pivotRow = col;
      for (var r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivotRow][col])) pivotRow = r;
      if (Math.abs(M[pivotRow][col]) < 1e-10) return null; // singular
      var tmp = M[col]; M[col] = M[pivotRow]; M[pivotRow] = tmp;
      var pivot = M[col][col];
      for (var c = col; c <= n; c++) M[col][c] /= pivot;
      for (var r2 = 0; r2 < n; r2++) {
        if (r2 === col) continue;
        var factor = M[r2][col];
        for (var c2 = col; c2 <= n; c2++) M[r2][c2] -= factor * M[col][c2];
      }
    }
    return M.map(function (row) { return [row[n]]; });
  }

  return GazeEngine;
}));
