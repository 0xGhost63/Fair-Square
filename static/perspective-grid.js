/**
 * PerspectiveGrid – React Bits Pro WebGL implementation for Flask
 * Config: color #d685d3 | speed 0.4 | gridScale 1.0 | lineThickness 1.0
 *         gridLength 10.0 | antialiasQuality 64 | perspective 0° | curve 0.0
 *         fadeSmoothness 1.00 | opacity 1.00 | autoPlay true
 */
(function () {
  var canvas = document.getElementById('perspective-grid-canvas');
  if (!canvas) return;

  var gl = canvas.getContext('webgl', { antialias: true, alpha: true, depth: false })
         || canvas.getContext('experimental-webgl', { antialias: true, alpha: true, depth: false });
  if (!gl) { console.warn('PerspectiveGrid: WebGL not available'); return; }

  /* Enable derivatives extension for fwidth (needed in GLSL ES 1.0) */
  var ext = gl.getExtension('OES_standard_derivatives');

  /* ---- Vertex shader ---- */
  var vsSource = [
    'attribute vec2 aPos;',
    'varying vec2 vUv;',
    'void main(){',
    '  vUv = aPos * 0.5 + 0.5;',
    '  gl_Position = vec4(aPos, 0.0, 1.0);',
    '}'
  ].join('\n');

  /* ---- Fragment shader ---- */
  var fsSource = [
    ext ? '#extension GL_OES_standard_derivatives : enable' : '',
    'precision highp float;',
    'varying vec2 vUv;',
    'uniform vec2  uRes;',
    'uniform float uTime;',
    'uniform vec3  uColor;',
    'uniform float uOpacity;',
    'uniform float uGridScale;',
    'uniform float uLineThick;',
    'uniform float uGridLen;',
    'uniform float uAA;',
    'uniform float uFadeSmooth;',
    'uniform float uCurve;',

    'void main(){',
    /* Discard upper half — pure black sky above horizon */
    '  float hor = 0.50;',
    '  if(vUv.y > hor){ gl_FragColor = vec4(0.0); return; }',

    /* Depth from horizon to bottom; 0 at horizon, 1 at bottom */
    '  float depth = (hor - vUv.y) / hor;',
    '  if(depth < 0.001){ gl_FragColor = vec4(0.0); return; }',

    /* Perspective-project XY → world XZ */
    '  float aspect = uRes.x / uRes.y;',
    '  float z  = 1.0 / (depth * uGridLen * 0.25);',
    '  float xc = (vUv.x - 0.5) * aspect;',
    '  float curve = uCurve * xc * xc;',
    '  float gx = xc * z;',
    '  float gy = z + uTime + curve;',

    /* Scale by gridScale */
    '  gx *= uGridScale * 1.8;',
    '  gy *= uGridScale * 1.8;',

    /* Anti-aliased grid lines */
    '  vec2 gv = vec2(gx, gy);',
    '  vec2 gf = abs(fract(gv - 0.5) - 0.5);', // distance to nearest line

    /* AA width: use derivatives if available, else fallback */
    ext
      ? '  vec2 dg = fwidth(gv);'
      : '  vec2 dg = vec2(0.02, 0.02);',
    '  float aaW = max(max(dg.x, dg.y), 1.0/uAA) * uLineThick * 1.4;',

    '  float lx = smoothstep(aaW, 0.0, gf.x);',
    '  float ly = smoothstep(aaW, 0.0, gf.y);',
    '  float grid = max(lx, ly);',

    /* Fade toward horizon (smooth) and at bottom edge */
    '  float fade   = smoothstep(0.0, 0.5 * uFadeSmooth, depth);',
    '  float bfade  = smoothstep(0.0, 0.12, vUv.y);',

    /* Subtle neon glow near horizon */
    '  float glow   = exp(-depth * 9.0) * 0.4;',
    '  vec3 col     = uColor + uColor * glow;',

    '  float alpha  = grid * fade * bfade * uOpacity;',
    '  gl_FragColor = vec4(col, alpha);',
    '}'
  ].join('\n');

  /* ---- Compile helpers ---- */
  function makeShader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('PerspectiveGrid shader error:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  var vs = makeShader(gl.VERTEX_SHADER, vsSource);
  var fs = makeShader(gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) return;

  var prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('PerspectiveGrid link error:', gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  /* Full-screen quad */
  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1,-1,  1,-1,  -1,1,
    -1, 1,  1,-1,   1,1
  ]), gl.STATIC_DRAW);
  var posLoc = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  /* Uniform locations */
  var uRes        = gl.getUniformLocation(prog, 'uRes');
  var uTime       = gl.getUniformLocation(prog, 'uTime');
  var uColor      = gl.getUniformLocation(prog, 'uColor');
  var uOpacity    = gl.getUniformLocation(prog, 'uOpacity');
  var uGridScale  = gl.getUniformLocation(prog, 'uGridScale');
  var uLineThick  = gl.getUniformLocation(prog, 'uLineThick');
  var uGridLen    = gl.getUniformLocation(prog, 'uGridLen');
  var uAA         = gl.getUniformLocation(prog, 'uAA');
  var uFadeSmooth = gl.getUniformLocation(prog, 'uFadeSmooth');
  var uCurve      = gl.getUniformLocation(prog, 'uCurve');

  /* Config matching screenshot settings exactly */
  var CFG = {
    color:        [214/255, 133/255, 211/255],  // #d685d3
    speed:        0.4,
    opacity:      1.0,
    gridScale:    1.0,
    lineThick:    1.0,
    gridLen:      10.0,
    aa:           64.0,
    fadeSmooth:   1.0,
    curve:        0.0
  };

  /* Resize canvas to fill viewport */
  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width  = Math.floor(window.innerWidth  * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener('resize', resize);
  resize();

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  var t0 = performance.now();

  function frame(now) {
    var elapsed = (now - t0) / 1000.0;
    var timeVal = elapsed * CFG.speed;

    gl.clearColor(0,0,0,0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.uniform2f(uRes,        canvas.width, canvas.height);
    gl.uniform1f(uTime,       timeVal);
    gl.uniform3fv(uColor,     CFG.color);
    gl.uniform1f(uOpacity,    CFG.opacity);
    gl.uniform1f(uGridScale,  CFG.gridScale);
    gl.uniform1f(uLineThick,  CFG.lineThick);
    gl.uniform1f(uGridLen,    CFG.gridLen);
    gl.uniform1f(uAA,         CFG.aa);
    gl.uniform1f(uFadeSmooth, CFG.fadeSmooth);
    gl.uniform1f(uCurve,      CFG.curve);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
