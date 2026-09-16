import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============ GLOBAL STATE ============
const state = {
    ws: null,
    connected: false,
    running: false,
    paused: false,
    step: 0,
    T_STEPS: 1200,
    speed: 1.0,
    neurons: [],
    neuronMap: new Map(),
    spikes: new Float32Array(),
    popRateHistory: [],
    motRateHistory: [],
    motorPoolHistory: { legL: [], legR: [], desc: [], t1: [], t2: [], t3: [] },
    rasterData: [],
    bounds: [0, 0],
    selectedNeuron: null,
    viewMode: 'all',
    animationId: null,
    lastFrame: 0,
    fps: 0,
    frameCount: 0,
    fpsLastUpdate: 0
};

// ============ THREE.JS SETUP ============
const canvas = document.getElementById('brain-canvas');
const viewportStack = document.getElementById('viewport-stack');
const view3D = document.querySelector('.viewport-3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050508, 1);

const scene = new THREE.Scene();
// Fog removed to prevent scene blackout

const camera = new THREE.PerspectiveCamera(45, 1, 10, 1000000);
camera.position.set(0, 0, 180000); // far default until centroids load and fit runs

// Fit the whole brain (X/Y spans) inside the frustum with margin.
// Called on data load and via Center — never per-frame so orbiting is free.
function fitCameraToBrain(resetView = true, margin = 1.3) {
    const sx = state.brainSpan ? state.brainSpan.sx : 91200;
    const sy = state.brainSpan ? state.brainSpan.sy : 64238;
    const t = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const aspect = camera.aspect || 1;
    const d = Math.max((sy / 2) / t, (sx / 2) / (t * aspect)) * margin;
    if (resetView) {
        controls.target.set(0, 0, 0);
        camera.position.set(0, 0, d);
    } else {
        const dir = camera.position.clone().sub(controls.target);
        if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
        dir.normalize();
        camera.position.copy(controls.target).addScaledVector(dir, d);
    }
    camera.lookAt(controls.target);
    controls.update();
}

function sizeThreeToContainer() {
    const rect = view3D ? view3D.getBoundingClientRect() : { width: window.innerWidth - 380, height: window.innerHeight };
    const w = Math.max(50, rect.width);
    const h = Math.max(50, rect.height);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
}

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 1000;
controls.maxDistance = 600000;
controls.autoRotate = false;
controls.autoRotateSpeed = 0.25;

// Lights
const ambient = new THREE.AmbientLight(0x666688, 0.7);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(50000, 50000, 50000);
scene.add(dirLight);

// ============ NEURON VISUALIZATION (CENTROIDS) ============
let neuronPoints = null;
let neuronGeometry = null;
let neuronMaterial = null;

function createSomaTexture() {
    const texCanvas = document.createElement('canvas');
    texCanvas.width = 64;
    texCanvas.height = 64;
    const ctx = texCanvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0.0, 'rgba(255, 255, 255, 0.8)');
    grad.addColorStop(0.2, 'rgba(230, 240, 255, 0.7)');
    grad.addColorStop(0.5, 'rgba(160, 200, 255, 0.35)');
    grad.addColorStop(0.8, 'rgba(80, 120, 255, 0.1)');
    grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(32, 32, 32, 0, Math.PI * 2);
    ctx.fill();
    return new THREE.CanvasTexture(texCanvas);
}

function createNeuronPoints(neurons) {
    if (!neurons || neurons.length === 0) return;

    if (neuronPoints) {
        scene.remove(neuronPoints);
        if (neuronGeometry) neuronGeometry.dispose();
    }

    state.neuronMap.clear();
    const count = neurons.length;
    neuronGeometry = new THREE.BufferGeometry();

    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const baseColors = new Float32Array(count * 3);
    state.spikeIntensity = new Float32Array(count);

    // Compute bounding box to dynamically center the fly brain centroids
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
        const n = neurons[i];
        if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
        if (n.z < minZ) minZ = n.z; if (n.z > maxZ) maxZ = n.z;
    }
    const cx = (minX + maxX) / 2 || 48000;
    const cy = (minY + maxY) / 2 || 37000;
    const cz = (minZ + maxZ) / 2 || 72000;
    state.brainSpan = {
        sx: (maxX - minX) || 91200,
        sy: (maxY - minY) || 64238,
        sz: (maxZ - minZ) || 124000
    };

    const colorMap = {
        'sensory': new THREE.Color(0x00d4ff),
        'interneuron': new THREE.Color(0xa78bfa),
        'motor': new THREE.Color(0xfb923c),
        'cb_intrinsic': new THREE.Color(0x38bdf8),
        'ol_intrinsic': new THREE.Color(0xf472b6),
        'vnc_intrinsic': new THREE.Color(0x4ade80),
        'visual_projection': new THREE.Color(0xfacc15),
        'vnc_sensory': new THREE.Color(0x2dd4bf),
        'ol_sensory': new THREE.Color(0xc084fc),
        'cb_sensory': new THREE.Color(0x34d399),
        'ascending_neuron': new THREE.Color(0xfde047),
        'descending_neuron': new THREE.Color(0xe879f9),
        'vnc_motor': new THREE.Color(0xf97316),
        'visual_centrifugal': new THREE.Color(0x22d3ee),
        'cb_motor': new THREE.Color(0xf43f5e),
    };

    // Dull background colors (desaturated)
    function dullColor(color, factor = 0.25) {
        const dull = color.clone();
        dull.r = 0.5 + (color.r - 0.5) * factor;
        dull.g = 0.5 + (color.g - 0.5) * factor;
        dull.b = 0.5 + (color.b - 0.5) * factor;
        return dull;
    }

    for (let i = 0; i < count; i++) {
        const n = neurons[i];
        const idx = i * 3;
        // Center the neuron centroids around (0, 0, 0)
        positions[idx] = n.x - cx;
        positions[idx + 1] = -(n.y - cy); // Invert Y so dorsal is upright
        positions[idx + 2] = n.z - cz;

        const type = n.superclass || n.class || n.type || 'interneuron';
        let color = colorMap[type] || colorMap.interneuron;
        
        // Make background neurons dull
        if (n.is_background) {
            color = dullColor(color, 0.15);
        }
        
        colors[idx] = color.r;
        colors[idx + 1] = color.g;
        colors[idx + 2] = color.b;

        baseColors[idx] = color.r;
        baseColors[idx + 1] = color.g;
        baseColors[idx + 2] = color.b;

        // Only monitored neurons (mon_index >= 0) receive spikes.
        // Background neurons use universe indices that numerically overlap
        // monitor positions, so they must NOT share this map.
        if (n.mon_index !== undefined && n.mon_index >= 0) {
            state.neuronMap.set(n.mon_index, i);
        }
    }

    state.baseColors = baseColors;

    neuronGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    neuronGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const somaTex = createSomaTexture();
    neuronMaterial = new THREE.PointsMaterial({
        size: 4.0,
        map: somaTex,
        vertexColors: true,
        transparent: true,
        opacity: 0.75,
        sizeAttenuation: false, // Ensures glowing points are always crisp and visible
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    neuronPoints = new THREE.Points(neuronGeometry, neuronMaterial);
    neuronPoints.frustumCulled = false;
    scene.add(neuronPoints);

    // Orientation axes
    const axesHelper = new THREE.AxesHelper(30000);
    scene.add(axesHelper);

    controls.target.set(0, 0, 0);
    controls.update();
    sizeThreeToContainer();
    fitCameraToBrain(true); // zoom to include the entire brain
    buildTopdownProjection();
}

function updateNeuronSpikes(spikes) {
    if (!neuronPoints || !neuronGeometry || !state.baseColors) return;
    const colors = neuronGeometry.attributes.color.array;
    let updated = false;

    for (let i = 0; i < spikes.length; i++) {
        if (spikes[i] > 0) {
            const neuronIdx = state.neuronMap.get(i);
            if (neuronIdx !== undefined) {
                const cIdx = neuronIdx * 3;
                // Monitored neurons: dimmer gold flash on spike (was 1.0/1.0/0.5)
                colors[cIdx] = 0.85;
                colors[cIdx + 1] = 0.8;
                colors[cIdx + 2] = 0.4;
                state.spikeIntensity[neuronIdx] = 0.7;
                updated = true;
            }
        }
    }
    if (updated) {
        neuronGeometry.attributes.color.needsUpdate = true;
    }
}

// ============ TOP-DOWN (DORSAL) PROJECTION ============
// Orthographic dorsal view: X -> horizontal, Z -> vertical (looking down Y).
// Shares state.neurons / state.neuronMap / state.spikeIntensity with the 3D
// view, so every gold flash in 3D appears in the projection too.
const topCanvas = document.getElementById('topdown-canvas');
const topCtx = topCanvas ? topCanvas.getContext('2d') : null;
const topStatic = document.createElement('canvas');
const topStaticCtx = topStatic.getContext('2d');
let topW = 0, topH = 0;
let topProj = null; // { px: Float32Array, py: Float32Array, css: string[], minX, maxX, minZ, maxZ }

function topdownColorFor(n) {
    const type = n.superclass || n.class || n.type || 'interneuron';
    const hexMap = {
        'sensory': '#00d4ff', 'interneuron': '#a78bfa', 'motor': '#fb923c',
        'cb_intrinsic': '#38bdf8', 'ol_intrinsic': '#f472b6', 'vnc_intrinsic': '#4ade80',
        'visual_projection': '#facc15', 'vnc_sensory': '#2dd4bf', 'ol_sensory': '#c084fc',
        'cb_sensory': '#34d399', 'ascending_neuron': '#fde047', 'descending_neuron': '#e879f9',
        'vnc_motor': '#f97316', 'visual_centrifugal': '#22d3ee', 'cb_motor': '#f43f5e',
    };
    let hex = hexMap[type] || hexMap.interneuron;
    if (n.is_background) {
        // Desaturate toward grey, matching dullColor(factor 0.15) in 3D
        const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
        const f = 0.15;
        const dr = Math.round(128 + (r - 128) * f), dg = Math.round(128 + (g - 128) * f), db = Math.round(128 + (b - 128) * f);
        hex = '#' + [dr, dg, db].map(v => v.toString(16).padStart(2, '0')).join('');
    }
    // Dim the static base layer so spike flashes stand out (top-down only;
    // the 3D view uses its own colors). Spikes keep full brightness.
    const r0 = parseInt(hex.slice(1, 3), 16), g0 = parseInt(hex.slice(3, 5), 16), b0 = parseInt(hex.slice(5, 7), 16);
    const dim = 0.42;
    return '#' + [r0, g0, b0].map(v => Math.round(v * dim).toString(16).padStart(2, '0')).join('');
}

function buildTopdownProjection() {
    if (!state.neurons || state.neurons.length === 0) return;
    const count = state.neurons.length;
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    let valid = 0;
    for (let i = 0; i < count; i++) {
        const n = state.neurons[i];
        if (!Number.isFinite(n.x) || !Number.isFinite(n.z)) continue;
        valid++;
        if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
        if (n.z < minZ) minZ = n.z; if (n.z > maxZ) maxZ = n.z;
    }
    if (!valid || !Number.isFinite(minX) || !Number.isFinite(minZ)) {
        console.warn('Top-down: no valid x/z centroids found');
        const st = document.getElementById('topdown-status');
        if (st) st.textContent = 'No valid x/z positions in data';
        return;
    }
    const px = new Float32Array(count), py = new Float32Array(count);
    const css = new Array(count);
    for (let i = 0; i < count; i++) {
        const n = state.neurons[i];
        px[i] = n.x; py[i] = n.z;
        css[i] = topdownColorFor(n);
    }
    topProj = { px, py, css, minX, maxX, minZ, maxZ, count };
    console.log(`Top-down: ${count} centroids (${valid} valid), X:[${minX.toFixed(0)},${maxX.toFixed(0)}] Z:[${minZ.toFixed(0)},${maxZ.toFixed(0)}]`);
    const st = document.getElementById('topdown-status');
    if (st) st.textContent = `${count.toLocaleString()} centroids · dorsal X–Z`;
    // Canvas may not have been measurable at first build — (re)size then paint.
    if (!topW || !topH) resizeTopdown();
    else renderTopdownStatic();
}

function topProjectToScreen(i, w, h) {
    // Fit with padding, preserve aspect. Rotated 180°: brain on top,
    // VNC at bottom (anterior/+Z down, X mirrored).
    const pad = 24;
    const spanX = Math.max(1, topProj.maxX - topProj.minX);
    const spanZ = Math.max(1, topProj.maxZ - topProj.minZ);
    const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanZ);
    const ox = (w - spanX * scale) / 2, oy = (h - spanZ * scale) / 2;
    return [
        ox + (topProj.maxX - topProj.px[i]) * scale,
        oy + (topProj.py[i] - topProj.minZ) * scale, // 180°: anterior (+Z) at bottom, X mirrored
        scale
    ];
}

function resizeTopdown() {
    if (!topCanvas || !topCtx) return;
    const rect = topCanvas.parentElement.getBoundingClientRect();
    // Parent may report 0 if layout hasn't settled or viewport is hidden —
    // keep last good size and let the watchdog retry.
    if (rect.width < 10 || rect.height < 10) return;
    // Parent is .viewport (flex child); canvas fills it below the label.
    const w = Math.max(50, rect.width), h = Math.max(50, rect.height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    topW = w; topH = h;
    topCanvas.width = Math.round(w * dpr);
    topCanvas.height = Math.round(h * dpr);
    topCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    topStatic.width = topCanvas.width;
    topStatic.height = topCanvas.height;
    topStaticCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderTopdownStatic();
}

function renderTopdownStatic() {
    if (!topProj || !topW || !topH) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    topStaticCtx.clearRect(0, 0, topW, topH);
    topStaticCtx.fillStyle = '#050508';
    topStaticCtx.fillRect(0, 0, topW, topH);
    // Faint midline + outline grid
    topStaticCtx.strokeStyle = '#1a1d24';
    topStaticCtx.lineWidth = 1;
    for (let gx = 0; gx < topW; gx += 48) {
        topStaticCtx.beginPath(); topStaticCtx.moveTo(gx, 0); topStaticCtx.lineTo(gx, topH); topStaticCtx.stroke();
    }
    for (let gy = 0; gy < topH; gy += 48) {
        topStaticCtx.beginPath(); topStaticCtx.moveTo(0, gy); topStaticCtx.lineTo(topW, gy); topStaticCtx.stroke();
    }
    // Static dots: fast fillRect per neuron (background layer).
    // Skipped entirely if a centroid has non-finite coords.
    const isSmall = topProj.count > 20000;
    const s = isSmall ? 2.2 : 3.0;
    for (let i = 0; i < topProj.count; i++) {
        if (!Number.isFinite(topProj.px[i]) || !Number.isFinite(topProj.py[i])) continue;
        const [sx, sy] = topProjectToScreen(i, topW, topH);
        if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;
        topStaticCtx.globalAlpha = 0.30;
        topStaticCtx.fillStyle = topProj.css[i];
        topStaticCtx.fillRect(sx - s / 2, sy - s / 2, s, s);
    }
    topStaticCtx.globalAlpha = 1;
    void dpr;
}

function drawTopdown() {
    if (!topCtx || !topW || !topH) return;
    if (!topProj) {
        // Data not yet loaded — placeholder instead of a black hole.
        topCtx.clearRect(0, 0, topW, topH);
        topCtx.fillStyle = '#050508';
        topCtx.fillRect(0, 0, topW, topH);
        topCtx.fillStyle = '#666';
        topCtx.font = '12px monospace';
        topCtx.textAlign = 'center';
        topCtx.fillText('Loading centroids… (waiting for /api/annotations)', topW / 2, topH / 2);
        topCtx.textAlign = 'left';
        return;
    }
    topCtx.clearRect(0, 0, topW, topH);
    topCtx.drawImage(topStatic, 0, 0, topW, topH);
    // Overlay spiking neurons as bright dots (same spikeIntensity as 3D,
    // same gold hue). Base layer is dimmed so flashes pop at full brightness.
    // Perf: solid fillRect, no per-spike gradients; capped + higher threshold.
    if (!state.spikeIntensity) return;
    let drawn = 0;
    for (let i = 0; i < topProj.count && drawn < 1500; i++) {
        const inten = state.spikeIntensity[i];
        if (inten > 0.18) {
            const [sx, sy] = topProjectToScreen(i, topW, topH);
            if (sx < -8 || sy < -8 || sx > topW + 8 || sy > topH + 8) continue;
            const s = 2 + inten * 2;
            topCtx.globalAlpha = 0.55 + 0.45 * inten;
            topCtx.fillStyle = '#ffd778';
            topCtx.fillRect(sx - s / 2, sy - s / 2, s, s);
            drawn++;
        }
    }
    topCtx.globalAlpha = 1;
    // Spike counter readout
    if (state.lastSpikeCount > 0) {
        topCtx.fillStyle = 'rgba(0,255,136,0.9)';
        topCtx.font = '11px monospace';
        topCtx.fillText(`spikes: ${state.lastSpikeCount}   t=${state.step}ms`, 10, topH - 10);
    }
}


// ============ RASTER CANVAS ============
const rasterCanvas = document.getElementById('raster-canvas');
const rasterCtx = rasterCanvas.getContext('2d');
let rasterWidth = 0, rasterHeight = 0;

function resizeRaster() {
    const rect = rasterCanvas.parentElement.getBoundingClientRect();
    rasterWidth = rect.width;
    rasterHeight = rect.height;
    rasterCanvas.width = rasterWidth * window.devicePixelRatio;
    rasterCanvas.height = rasterHeight * window.devicePixelRatio;
    rasterCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function drawRaster(spikes, bounds) {
    if (!rasterCtx) return;
    
    const w = rasterWidth;
    const h = rasterHeight;
    const history = state.rasterData;
    const maxHistory = Math.floor(w / 2);
    
    // Shift history
    history.unshift({ spikes, bounds });
    if (history.length > maxHistory) history.pop();
    
    rasterCtx.fillStyle = '#0a0a0f';
    rasterCtx.fillRect(0, 0, w, h);
    
    // Draw grid lines
    rasterCtx.strokeStyle = '#1a1d24';
    rasterCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
        rasterCtx.beginPath();
        rasterCtx.moveTo(x, 0);
        rasterCtx.lineTo(x, h);
        rasterCtx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
        rasterCtx.beginPath();
        rasterCtx.moveTo(0, y);
        rasterCtx.lineTo(w, y);
        rasterCtx.stroke();
    }
    
    // Draw spikes
    const neuronCount = spikes.length;
    const yScale = h / neuronCount;
    
    history.forEach((frame, frameIdx) => {
        const x = w - frameIdx * 2 - 2;
        if (x < 0) return;
        
        frame.spikes.forEach((spike, nIdx) => {
            if (spike > 0) {
                const y = nIdx * yScale + yScale / 2;
                let color = '#8888ff';
                if (nIdx < frame.bounds[0]) color = '#00aaff';
                else if (nIdx < frame.bounds[1]) color = '#aa88ff';
                else color = '#ffaa00';
                
                rasterCtx.fillStyle = color;
                rasterCtx.fillRect(x, y - 1, 2, 2);
            }
        });
    });
    
    // Draw boundary lines
    rasterCtx.strokeStyle = '#ff4444';
    rasterCtx.lineWidth = 1;
    rasterCtx.setLineDash([4, 4]);
    bounds.forEach(b => {
        const y = b * yScale;
        rasterCtx.beginPath();
        rasterCtx.moveTo(0, y);
        rasterCtx.lineTo(w, y);
        rasterCtx.stroke();
    });
    rasterCtx.setLineDash([]);
}

// ============ RATE CHART ============
const rateCanvas = document.getElementById('rate-chart');
const rateCtx = rateCanvas.getContext('2d');
let rateWidth = 0, rateHeight = 0;

function resizeRateChart() {
    const rect = rateCanvas.parentElement.getBoundingClientRect();
    rateWidth = rect.width;
    rateHeight = rect.height;
    rateCanvas.width = rateWidth * window.devicePixelRatio;
    rateCanvas.height = rateHeight * window.devicePixelRatio;
    rateCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function drawRateChart() {
    if (!rateCtx) return;
    
    const w = rateWidth;
    const h = rateHeight;
    const pop = state.popRateHistory;
    const mot = state.motRateHistory;
    const maxLen = Math.min(pop.length, mot.length, w);
    
    rateCtx.fillStyle = '#0a0a0f';
    rateCtx.fillRect(0, 0, w, h);
    
    // Grid
    rateCtx.strokeStyle = '#1a1d24';
    rateCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
        rateCtx.beginPath();
        rateCtx.moveTo(x, 0);
        rateCtx.lineTo(x, h);
        rateCtx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
        rateCtx.beginPath();
        rateCtx.moveTo(0, y);
        rateCtx.lineTo(w, y);
        rateCtx.stroke();
    }
    
    // Stim windows
    rateCtx.fillStyle = 'rgba(255, 170, 0, 0.1)';
    const stimWindows = [[300, 500], [700, 900]];
    stimWindows.forEach(([a, b]) => {
        const x1 = (a / state.T_STEPS) * w;
        const x2 = (b / state.T_STEPS) * w;
        rateCtx.fillRect(x1, 0, x2 - x1, h);
    });
    
    // Current step indicator
    rateCtx.strokeStyle = '#00ff88';
    rateCtx.lineWidth = 1;
    rateCtx.setLineDash([4, 4]);
    const cx = (state.step / state.T_STEPS) * w;
    rateCtx.beginPath();
    rateCtx.moveTo(cx, 0);
    rateCtx.lineTo(cx, h);
    rateCtx.stroke();
    rateCtx.setLineDash([]);
    
    // Pop rate
    rateCtx.strokeStyle = '#00ff88';
    rateCtx.lineWidth = 2;
    rateCtx.beginPath();
    for (let i = 0; i < maxLen; i++) {
        const x = (i / maxLen) * w;
        const y = h - (pop[pop.length - maxLen + i] || 0) * h * 5;
        if (i === 0) rateCtx.moveTo(x, y);
        else rateCtx.lineTo(x, y);
    }
    rateCtx.stroke();
    
    // Motor rate
    rateCtx.strokeStyle = '#ffaa00';
    rateCtx.lineWidth = 2;
    rateCtx.beginPath();
    for (let i = 0; i < maxLen; i++) {
        const x = (i / maxLen) * w;
        const y = h - (mot[mot.length - maxLen + i] || 0) * h * 10;
        if (i === 0) rateCtx.moveTo(x, y);
        else rateCtx.lineTo(x, y);
    }
    rateCtx.stroke();
}

// ============ MOTOR POOL CHART ============
const MOTOR_POOLS = [
    { key: 'legL', label: 'Leg L', color: '#00aaff' },
    { key: 'legR', label: 'Leg R', color: '#ff66aa' },
    { key: 'desc', label: 'Desc', color: '#b48cff' },
    { key: 't1', label: 'T1', color: '#ffd166' },
    { key: 't2', label: 'T2', color: '#06d6a0' },
    { key: 't3', label: 'T3', color: '#ef476f' },
];
const MOTOR_SCALE = 50; // pool fractions are ~0-0.02, amplify to fill chart
const motorCanvas = document.getElementById('motor-chart');
const motorCtx = motorCanvas ? motorCanvas.getContext('2d') : null;
let motorWidth = 0, motorHeight = 0;

function resetMotorPools() {
    MOTOR_POOLS.forEach(p => { state.motorPoolHistory[p.key] = []; });
}

function resizeMotorChart() {
    if (!motorCanvas || !motorCtx) return;
    const rect = motorCanvas.parentElement.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    motorWidth = rect.width;
    motorHeight = rect.height;
    motorCanvas.width = motorWidth * window.devicePixelRatio;
    motorCanvas.height = motorHeight * window.devicePixelRatio;
    motorCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
}

function drawMotorChart() {
    if (!motorCtx || motorWidth < 2) return;
    const w = motorWidth, h = motorHeight;

    motorCtx.fillStyle = '#0a0a0f';
    motorCtx.fillRect(0, 0, w, h);

    // Grid
    motorCtx.strokeStyle = '#1a1d24';
    motorCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
        motorCtx.beginPath();
        motorCtx.moveTo(x, 0);
        motorCtx.lineTo(x, h);
        motorCtx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
        motorCtx.beginPath();
        motorCtx.moveTo(0, y);
        motorCtx.lineTo(w, y);
        motorCtx.stroke();
    }

    let maxLen = w;
    MOTOR_POOLS.forEach(p => {
        maxLen = Math.min(maxLen, state.motorPoolHistory[p.key].length);
    });
    if (maxLen < 2) return;

    MOTOR_POOLS.forEach(p => {
        const hist = state.motorPoolHistory[p.key];
        motorCtx.strokeStyle = p.color;
        motorCtx.lineWidth = 1.5;
        motorCtx.beginPath();
        for (let i = 0; i < maxLen; i++) {
            const v = hist[hist.length - maxLen + i] || 0;
            const x = (i / maxLen) * w;
            const y = h - Math.min(1, v * MOTOR_SCALE) * (h - 4) - 2;
            if (i === 0) motorCtx.moveTo(x, y);
            else motorCtx.lineTo(x, y);
        }
        motorCtx.stroke();
    });
}

// ============ WEBSOCKET ============
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    state.ws = new WebSocket(wsUrl);
    
    state.ws.onopen = () => {
        state.connected = true;
        updateConnectionStatus();
        console.log('WebSocket connected');
    };
    
    state.ws.onclose = () => {
        state.connected = false;
        updateConnectionStatus();
        setTimeout(connectWebSocket, 2000);
    };
    
    state.ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
    
    state.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'init':
            state.T_STEPS = msg.state.T_STEPS || 1200;
            state.motorLeftCount = msg.state.motor_left_count;
            state.motorRightCount = msg.state.motor_right_count;
            updateStats(msg.state);
            if (!state.neurons || state.neurons.length === 0) {
                fetchCentroids();
            }
            break;
        case 'status':
            if (msg.running !== undefined) state.running = msg.running;
            if (msg.paused !== undefined) state.paused = msg.paused;
            if (msg.step !== undefined) state.step = msg.step;
            updateSimulationStatus();
            break;
        case 'spikes':
            state.step = msg.t;
            state.spikes = new Float32Array(msg.spikes);
            state.popRateHistory.push(msg.pop_rate);
            state.motRateHistory.push(msg.mot_rate);
            if (state.popRateHistory.length > 2000) state.popRateHistory.shift();
            if (state.motRateHistory.length > 2000) state.motRateHistory.shift();
            if (msg.motor) {
                MOTOR_POOLS.forEach(p => {
                    const hist = state.motorPoolHistory[p.key];
                    hist.push(msg.motor[p.key] || 0);
                    if (hist.length > 2000) hist.shift();
                });
            }

            state.bounds = msg.bounds;
            const nSpikes = msg.n_spikes !== undefined
                ? msg.n_spikes
                : msg.spikes.reduce((a, b) => a + (b > 0 ? 1 : 0), 0);
            state.lastSpikeCount = nSpikes;
            if (nSpikes > 0) {
                console.log(`t=${msg.t} spikes=${nSpikes} pop=${(msg.pop_rate * 100).toFixed(2)}% mot=${(msg.mot_rate * 100).toFixed(2)}%`);
            }
            // Manual-burst frames are forced broadcasts — flash the viewport
            // edge so the stimulation is unmissable even at a glance.
            if (msg.stim_active) {
                flashStimOverlay();
            }
            updateNeuronSpikes(msg.spikes);
            drawRaster(msg.spikes, msg.bounds);
            drawRateChart();
            drawMotorChart();
            updateStats({ step: msg.step, pop_rate: msg.pop_rate, mot_rate: msg.mot_rate });
            break;

        case 'complete':
            state.running = false;
            state.popRateHistory = msg.pop_rate_history;
            state.motRateHistory = msg.mot_rate_history;
            state.rasterData = msg.raster;
            updateSimulationStatus();
            drawRateChart();
            break;
        case 'warning':
            console.warn(msg.message);
            showToast(msg.message, 'warning');
            break;
        case 'flygym_frame': {
            const img = document.getElementById('flygym-img');
            if (img && msg.jpg) img.src = 'data:image/jpeg;base64,' + msg.jpg;
            const st = document.getElementById('flygym-status');
            if (st) st.textContent = 'FlyGym · live · t=' + msg.step + 'ms';
            const tele = document.getElementById('flygym-tele');
            if (tele && msg.tele) {
                const drv = (msg.tele.ampL !== undefined)
                    ? ` · drv L/R ${((msg.tele.ampL || 0) * 100).toFixed(0)}/${((msg.tele.ampR || 0) * 100).toFixed(0)}% spd ${((msg.tele.spd || 0) * 100).toFixed(0)}%` : '';
                tele.textContent = `food ${msg.tele.food_dist}mm · touch ${(msg.tele.touch * 100).toFixed(0)}% · speed ${(msg.tele.speed * 100).toFixed(0)}%${drv} · fly (${msg.tele.fly_xy[0].toFixed(1)}, ${msg.tele.fly_xy[1].toFixed(1)})`;
            }
            break;
        }
    }
}

function sendControl(action) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: 'control', ...action }));
    }
}

// ============ UI UPDATES ============
function updateConnectionStatus() {
    const dot = document.getElementById('conn-dot');
    const status = document.getElementById('conn-status');
    if (state.connected) {
        dot.classList.add('connected');
        status.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        status.textContent = 'Disconnected';
    }
}

function updateSimulationStatus() {
    const dot = document.getElementById('sim-dot');
    const status = document.getElementById('sim-status');
    const runBtn = document.getElementById('btn-run');
    
    if (state.running && !state.paused) {
        dot.classList.add('running');
        status.textContent = 'Running';
        runBtn.textContent = 'Running';
        runBtn.disabled = true;
    } else if (state.paused) {
        dot.classList.remove('running');
        status.textContent = 'Paused';
        runBtn.textContent = 'Resume';
        runBtn.disabled = false;
    } else {
        dot.classList.remove('running');
        status.textContent = 'Stopped';
        runBtn.textContent = 'Run';
        runBtn.disabled = false;
    }
}

function updateStats(data) {
    if (data.num_neurons !== undefined) document.getElementById('stat-neurons').textContent = data.num_neurons.toLocaleString();
    if (data.num_synapses !== undefined) document.getElementById('stat-synapses').textContent = data.num_synapses.toLocaleString();
    if (data.num_sensory !== undefined) document.getElementById('stat-sensory').textContent = data.num_sensory.toLocaleString();
    if (data.num_motor !== undefined) document.getElementById('stat-motor').textContent = data.num_motor.toLocaleString();
    if (data.num_monitored !== undefined) document.getElementById('stat-monitored').textContent = data.num_monitored.toLocaleString();
    if (data.step !== undefined) document.getElementById('stat-step').textContent = data.step;
    if (data.pop_rate !== undefined) document.getElementById('stat-poprate').textContent = (data.pop_rate * 100).toFixed(2) + '%';
    if (data.mot_rate !== undefined) document.getElementById('stat-motrate').textContent = (data.mot_rate * 100).toFixed(2) + '%';
}

// ============ CONTROLS ============
function setupControls() {
    // Sliders
    const sliders = [
        { id: 'speed', valId: 'speed-val', suffix: 'x', action: 'set_speed', param: 'value' },
        { id: 'gain', valId: 'gain-val', suffix: '', action: 'set_param', param: 'gain' },
        { id: 'noise', valId: 'noise-val', suffix: '', action: 'set_param', param: 'noise' },
        { id: 'tau', valId: 'tau-val', suffix: '', action: 'set_param', param: 'tau_mem' },
        { id: 'thr', valId: 'thr-val', suffix: '', action: 'set_param', param: 'v_thr' },
        { id: 'stim-rate', valId: 'stim-rate-val', suffix: '', action: 'set_param', param: 'stim_rate_hz' },
        { id: 'stim-frac', valId: 'stim-frac-val', suffix: '', action: 'set_param', param: 'stim_fraction' },
        { id: 'i-ext', valId: 'i-ext-val', suffix: '', action: 'set_param', param: 'i_ext' }
    ];
    
    sliders.forEach(s => {
        const el = document.getElementById(s.id);
        const valEl = document.getElementById(s.valId);
        el.addEventListener('input', (e) => {
            const v = parseFloat(e.target.value);
            valEl.textContent = v.toFixed(s.suffix ? 1 : (s.id === 'noise' ? 2 : 0)) + s.suffix;
            if (s.action === 'set_param') {
                sendControl({ action: s.action, param: s.param, value: v });
            } else {
                sendControl({ action: s.action, [s.param]: v });
            }
        });
    });
    
    // Buttons
    document.getElementById('btn-run').addEventListener('click', () => {
        if (!state.running) {
            sendControl({ action: 'start' });
            state.running = true;
            state.paused = false;
            state.step = 0;
            state.popRateHistory = [];
            state.motRateHistory = [];
            resetMotorPools();
            state.rasterData = [];
            updateSimulationStatus();
        } else if (state.paused) {
            sendControl({ action: 'resume' });
            state.paused = false;
            updateSimulationStatus();
        }
    });
    
    document.getElementById('btn-pause').addEventListener('click', () => {
        if (state.running && !state.paused) {
            sendControl({ action: 'pause' });
            state.paused = true;
            updateSimulationStatus();
        }
    });
    
    document.getElementById('btn-reset').addEventListener('click', () => {
        sendControl({ action: 'reset' });
        state.running = false;
        state.paused = false;
        state.step = 0;
        state.popRateHistory = [];
        state.motRateHistory = [];
        resetMotorPools();
        state.rasterData = [];
        updateSimulationStatus();
        drawRateChart();
        drawMotorChart();
        resizeRaster();
        rasterCtx.fillStyle = '#0a0a0f';
        rasterCtx.fillRect(0, 0, rasterWidth, rasterHeight);
    });

    // Manual stimulation buttons (auto-starts a stopped/finished sim server-side)
    document.getElementById('btn-stimulate').addEventListener('click', () => {
        sendControl({ action: 'stimulate_sensory' });
        state.running = true;
        state.paused = false;
        updateSimulationStatus();
        showToast('Sensory stimulation triggered — watch for gold flashes', 'info');
    });

    document.getElementById('btn-stimulate-all').addEventListener('click', () => {
        sendControl({ action: 'stimulate_all' });
        state.running = true;
        state.paused = false;
        updateSimulationStatus();
        showToast('Global stimulation triggered — watch for gold flashes', 'info');
    });

    const flyFoodBtn = document.getElementById('btn-flyfood');
    if (flyFoodBtn) flyFoodBtn.addEventListener('click', () => {
        sendControl({ action: 'flygym_drop_food' });
        showToast('Berry dropped near the fly', 'info');
    });
    const flyResetBtn = document.getElementById('btn-flyreset');
    if (flyResetBtn) flyResetBtn.addEventListener('click', () => {
        sendControl({ action: 'flygym_reset' });
        showToast('FlyGym body reset', 'info');
    });
    
    // Presets
    const presets = {
        default: { gain: 2.5, noise: 0.06, tau_mem: 10, v_thr: 0.8, stim_rate_hz: 60, stim_fraction: 0.3, i_ext: 2.0 },
        'high-gain': { gain: 2.5, noise: 0.03, tau_mem: 10, v_thr: 1.0, stim_rate_hz: 40, stim_fraction: 0.25, i_ext: 1.5 },
        'low-noise': { gain: 1.6, noise: 0.005, tau_mem: 10, v_thr: 1.0, stim_rate_hz: 40, stim_fraction: 0.25, i_ext: 1.5 },
        burst: { gain: 2.0, noise: 0.05, tau_mem: 5, v_thr: 0.8, stim_rate_hz: 80, stim_fraction: 0.4, i_ext: 2.0 },
        quiet: { gain: 1.0, noise: 0.01, tau_mem: 20, v_thr: 1.5, stim_rate_hz: 20, stim_fraction: 0.1, i_ext: 1.0 },
        seizure: { gain: 3.0, noise: 0.1, tau_mem: 5, v_thr: 0.5, stim_rate_hz: 100, stim_fraction: 0.5, i_ext: 3.0 }
    };
    
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const preset = presets[btn.dataset.preset];
            Object.entries(preset).forEach(([param, value]) => {
                const sliderId = param.replace(/_/g, '-');
                const slider = document.getElementById(sliderId);
                const valEl = document.getElementById(sliderId + '-val');
                if (slider && valEl) {
                    slider.value = value;
                    valEl.textContent = value.toFixed(param === 'noise' ? 2 : (param.includes('rate') || param.includes('frac') ? 0 : 1)) + (param.includes('speed') ? 'x' : '');
                    sendControl({ action: 'set_param', param, value });
                }
            });
        });
    });
    
    // View mode buttons: all / split / 3d / topdown / fly
    document.querySelectorAll('[data-view]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.viewMode = btn.dataset.view;
            if (viewportStack) {
                viewportStack.classList.remove('mode-all', 'mode-split', 'mode-3d', 'mode-topdown', 'mode-fly');
                if (state.viewMode === '3d') viewportStack.classList.add('mode-3d');
                else if (state.viewMode === 'topdown') viewportStack.classList.add('mode-topdown');
                else if (state.viewMode === 'fly') viewportStack.classList.add('mode-fly');
                else if (state.viewMode === 'split') viewportStack.classList.add('mode-split');
                else viewportStack.classList.add('mode-all');
            }
            // Layout changed -> re-measure all viewports
            requestAnimationFrame(() => { onResize(); });
        });
    });
    
}

// ============ TOOLTIP & NEURON SELECTION ============
const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 1500 };
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');


function onMouseMove(event) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    if (!tooltip) return;
    tooltip.style.left = (event.clientX + 16) + 'px';
    tooltip.style.top = (event.clientY + 16) + 'px';
}

function onClick(event) {
    if (state.viewMode === 'topdown') return; // top-down has its own handler below

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(neuronPoints);
    
    if (intersects.length > 0) {
        const idx = intersects[0].index;
        const neuron = state.neurons[idx];
        if (neuron) {
            showNeuronDetail(neuron);
        }
    }
}

function showNeuronDetail(neuron) {
    state.selectedNeuron = neuron;
    const detail = document.getElementById('neuron-detail');
    detail.innerHTML = `
        <div style="color: #00ff88; font-size: 18px; font-weight: 600; margin-bottom: 8px;">Neuron #${neuron.index}</div>
        <div class="tooltip">
            <div class="row"><span class="key">Body ID:</span><span class="val">${neuron.id}</span></div>
            <div class="row"><span class="key">Type:</span><span class="val">${neuron.type}</span></div>
            <div class="row"><span class="key">Superclass:</span><span class="val">${neuron.superclass || 'N/A'}</span></div>
            <div class="row"><span class="key">Class:</span><span class="val">${neuron.class || 'N/A'}</span></div>
            <div class="row"><span class="key">Position:</span><span class="val">${(neuron.x || 0).toFixed(0)}, ${(neuron.y || 0).toFixed(0)}, ${(neuron.z || 0).toFixed(0)}</span></div>
        </div>
    `;
}

canvas.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('click', onClick);

// Top-down click-to-inspect: nearest projected neuron within 10px
if (topCanvas) {
    topCanvas.addEventListener('click', (event) => {
        if (!topProj || !state.neurons.length) return;
        const rect = topCanvas.getBoundingClientRect();
        const mx = event.clientX - rect.left, my = event.clientY - rect.top;
        let best = -1, bestD2 = 10 * 10;
        // Scan monitored-mapped neurons first for relevance, fall back to all
        const step = topProj.count > 20000 ? 2 : 1;
        for (let i = 0; i < topProj.count; i += step) {
            const [sx, sy] = topProjectToScreen(i, topW, topH);
            const dx = sx - mx, dy = sy - my;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; best = i; }
        }
        if (best >= 0 && state.neurons[best]) showNeuronDetail(state.neurons[best]);
    });
}

// ============ ANIMATION LOOP ============
function animate(time) {
    state.animationId = requestAnimationFrame(animate);
    
    // FPS counter
    state.frameCount++;
    if (time - state.fpsLastUpdate >= 1000) {
        state.fps = Math.round(state.frameCount * 1000 / (time - state.fpsLastUpdate));
        document.getElementById('fps-counter').textContent = `${state.fps} FPS`;
        state.frameCount = 0;
        state.fpsLastUpdate = time;
    }
    
    // Smooth spike glow decay (faster decay = dimmer + fewer buffer uploads)
    if (neuronGeometry && state.spikeIntensity && state.baseColors) {
        const colors = neuronGeometry.attributes.color.array;
        const baseColors = state.baseColors;
        let needsUpdate = false;
        for (let i = 0; i < state.spikeIntensity.length; i++) {
            if (state.spikeIntensity[i] > 0.02) {
                state.spikeIntensity[i] *= 0.9;
                const intensity = state.spikeIntensity[i];
                const cIdx = i * 3;
                colors[cIdx] = baseColors[cIdx] * (1.0 - intensity) + 0.85 * intensity;
                colors[cIdx + 1] = baseColors[cIdx + 1] * (1.0 - intensity) + 0.8 * intensity;
                colors[cIdx + 2] = baseColors[cIdx + 2] * (1.0 - intensity) + 0.4 * intensity;
                needsUpdate = true;
            }
        }
        if (needsUpdate) {
            neuronGeometry.attributes.color.needsUpdate = true;
        }
    }
    
    // Auto-rotate when not interacting (only when the brain is visible)
    if (!controls.userPan && !controls.userRotate &&
        (state.viewMode === '3d' || state.viewMode === 'split' || state.viewMode === 'all')) {
        controls.autoRotate = true;
    } else {
        controls.autoRotate = false;
    }

    controls.update();
    // Only pay for the 3D render when its viewport is visible
    if (state.viewMode !== 'topdown' && state.viewMode !== 'fly') {
        renderer.render(scene, camera);
    }
    if (state.viewMode !== 'fly') {
        drawTopdown();
    }
}

// ============ STIM FLASH OVERLAY ============
let stimFlashTimeout = null;
function flashStimOverlay() {
    canvas.style.boxShadow = 'inset 0 0 60px rgba(0, 255, 136, 0.55)';
    if (stimFlashTimeout) clearTimeout(stimFlashTimeout);
    stimFlashTimeout = setTimeout(() => { canvas.style.boxShadow = 'none'; }, 350);
}

// ============ RESIZE ============
function onResize() {
    sizeThreeToContainer();
    resizeTopdown();

    resizeRaster();
    resizeRateChart();
    resizeMotorChart();
    drawRateChart();
    drawMotorChart();
}

window.addEventListener('resize', onResize);

// ============ TOAST ============
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; z-index: 1000;
        background: ${type === 'warning' ? '#2a1010' : '#16181d'};
        border: 1px solid ${type === 'warning' ? '#ff6b6b' : '#333'};
        border-radius: 8px; padding: 12px 20px; color: ${type === 'warning' ? '#ff6b6b' : '#e0e0e0'};
        box-shadow: 0 4px 20px rgba(0,0,0,0.5); animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
`;
document.head.appendChild(style);

// ============ FETCH CENTROIDS & INIT ============
async function fetchCentroids() {
    try {
        const response = await fetch('/api/annotations');
        const data = await response.json();
        if (data && data.neurons && data.neurons.length > 0) {
            state.neurons = data.neurons;
            createNeuronPoints(state.neurons);
            console.log(`Loaded ${state.neurons.length} neuron centroids in 3D`);
            return true;
        }
    } catch (e) {
        console.warn('Could not load neuron centroids yet:', e);
    }
    return false;
}

async function init() {
    window.__flybrain_booted = true;
    setupControls();
    connectWebSocket();
    
    // Initial fetch of centroids
    const loaded = await fetchCentroids();
    if (!loaded) {
        // Poll every 1.5s until the connectome completes loading
        const pollInterval = setInterval(async () => {
            const ok = await fetchCentroids();
            if (ok) clearInterval(pollInterval);
        }, 1500);
    }
    
    onResize();
    resizeRaster();
    resizeRateChart();
    resizeMotorChart();
    resizeTopdown();

    // Slow state poll: keeps counts fresh. Cheap 2s GET, independent of the WS.
    setInterval(async () => {
        try {
            const r = await fetch('/api/state');
            if (r.ok) updateStats(await r.json());
        } catch (e) { /* server loading or down */ }
    }, 2000);

    // Watchdog: flex layout / late data can leave either viewport at 0px.
    // Re-measure cheaply once a second and repaint when size changes.
    setInterval(() => {
        if (view3D) {
            const r = view3D.getBoundingClientRect();
            if (r.width >= 10 && r.height >= 10 &&
                (Math.abs(r.width - renderer.domElement.clientWidth) > 2 ||
                 Math.abs(r.height - renderer.domElement.clientHeight) > 2)) {
                sizeThreeToContainer();
            }
        }
        if (topCanvas) {
            const r = topCanvas.parentElement.getBoundingClientRect();
            if (r.width >= 10 && r.height >= 10 &&
                (Math.abs(r.width - topW) > 2 || Math.abs(r.height - topH) > 2)) {
                resizeTopdown();
            }
        }
    }, 1000);

    animate(0);
}

init();