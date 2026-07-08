const canvas = document.getElementById('reveal-canvas');
const ctx = canvas.getContext('2d');

const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d');

let width, height;

// Note: Replace these URLs with the paths to your local images (e.g., 'aespa.jpg' and 'nasa.jpg')
const topImage = new Image();
// Using a placeholder image for the top layer
topImage.src = 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop'; 

const bottomImage = new Image();
// Using a placeholder image for the bottom layer (revealed)
bottomImage.src = 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2672&auto=format&fit=crop'; 

let imagesLoaded = 0;
topImage.onload = () => { imagesLoaded++; init(); };
bottomImage.onload = () => { imagesLoaded++; init(); };

let trail = [];
const MAX_AGE = 50; // How many frames a trail point lives
const MAX_RADIUS = 150; // The maximum size of the hole
let mouseX = -1000;
let mouseY = -1000;
let isMouseActive = false;
let lastMoveTime = 0;

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
}

window.addEventListener('resize', resize);
resize();

function init() {
    // Wait until both images are loaded before rendering
    if (imagesLoaded < 2) return;
    requestAnimationFrame(render);
}

// Track mouse/touch
function handleMove(x, y) {
    // If the mouse jumped a lot (e.g. fast movement), we could interpolate points here 
    // for a smoother trail, but pushing raw coordinates works for standard movement.
    mouseX = x;
    mouseY = y;
    isMouseActive = true;
    lastMoveTime = Date.now();
    trail.push({ x: x, y: y, age: 0 });
}

window.addEventListener('mousemove', (e) => {
    handleMove(e.clientX, e.clientY);
});

window.addEventListener('touchmove', (e) => {
    if(e.touches.length > 0) {
        handleMove(e.touches[0].clientX, e.touches[0].clientY);
    }
}, {passive: true});

window.addEventListener('mouseout', () => {
    isMouseActive = false;
});
window.addEventListener('touchend', () => {
    isMouseActive = false;
});

// Helper function to draw image similar to CSS background-size: cover
function drawCoverImage(context, img, cw, ch) {
    const imgRatio = img.width / img.height;
    const canvasRatio = cw / ch;
    let drawWidth, drawHeight, offsetX, offsetY;

    if (imgRatio > canvasRatio) {
        drawHeight = ch;
        drawWidth = img.width * (ch / img.height);
        offsetX = (cw - drawWidth) / 2;
        offsetY = 0;
    } else {
        drawWidth = cw;
        drawHeight = img.height * (cw / img.width);
        offsetX = 0;
        offsetY = (ch - drawHeight) / 2;
    }
    context.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
}

function render() {
    // 1. Draw Bottom Image directly to main canvas
    ctx.globalCompositeOperation = 'source-over';
    drawCoverImage(ctx, bottomImage, width, height);

    // 2. Prepare Offscreen Canvas with Top Image
    offscreenCtx.globalCompositeOperation = 'source-over';
    drawCoverImage(offscreenCtx, topImage, width, height);

    // 3. Punch Hole on Offscreen Canvas using destination-out
    offscreenCtx.globalCompositeOperation = 'destination-out';

    // Idle detection fades the entire trail out after 100ms of no movement
    const isIdle = (Date.now() - lastMoveTime) > 100;

    // Process and draw the trail
    for (let i = 0; i < trail.length; i++) {
        let p = trail[i];
        
        // Age the point. If idle, age it much faster to rapidly fade out the trail
        p.age += isIdle ? 4 : 1; 

        if (p.age > MAX_AGE) continue;

        // Tapering snake effect: size decreases as age increases
        const progress = p.age / MAX_AGE;
        // Using ease-out logic for a smoother taper
        const radius = Math.max(0, MAX_RADIUS * (1 - Math.pow(progress, 1.5))); 
        
        if (radius > 0) {
            // Feathered mask using radial gradient
            const gradient = offscreenCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
            gradient.addColorStop(0, 'rgba(0, 0, 0, 1)');    // Center fully cuts out
            gradient.addColorStop(0.4, 'rgba(0, 0, 0, 0.8)'); 
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');    // Edges feathered
            
            offscreenCtx.fillStyle = gradient;
            offscreenCtx.beginPath();
            offscreenCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            offscreenCtx.fill();
        }
    }

    // Clean up dead points
    trail = trail.filter(p => p.age <= MAX_AGE);

    // 4. Draw Offscreen canvas (Top Image with holes) onto the main canvas
    ctx.drawImage(offscreenCanvas, 0, 0);

    // 5. Draw custom reticle and ember glow over everything
    if (isMouseActive && mouseX >= 0 && mouseY >= 0) {
        ctx.globalCompositeOperation = 'source-over';
        
        // Ember glow ring
        const glowRadius = 30;
        const glowGradient = ctx.createRadialGradient(mouseX, mouseY, 5, mouseX, mouseY, glowRadius);
        glowGradient.addColorStop(0, 'rgba(255, 120, 50, 0.8)'); // Bright ember core
        glowGradient.addColorStop(0.5, 'rgba(255, 50, 0, 0.4)');
        glowGradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
        
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        // Reticle
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        
        // Center Circle
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, 6, 0, Math.PI * 2);
        ctx.stroke();
        
        // Outer Crosshairs
        const chOffset = 12;
        const chLength = 8;
        ctx.beginPath();
        // Left
        ctx.moveTo(mouseX - chOffset - chLength, mouseY);
        ctx.lineTo(mouseX - chOffset, mouseY);
        // Right
        ctx.moveTo(mouseX + chOffset, mouseY);
        ctx.lineTo(mouseX + chOffset + chLength, mouseY);
        // Top
        ctx.moveTo(mouseX, mouseY - chOffset - chLength);
        ctx.lineTo(mouseX, mouseY - chOffset);
        // Bottom
        ctx.moveTo(mouseX, mouseY + chOffset);
        ctx.lineTo(mouseX, mouseY + chOffset + chLength);
        ctx.stroke();
    }

    requestAnimationFrame(render);
}
