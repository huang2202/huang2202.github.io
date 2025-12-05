/* * 鼠标点击特效 - 透明气泡版 (Transparent Bubble Zoom)
 * 效果：马卡龙色系，中间半透明，边缘清晰，点击炸开后原地缩小消失
 */
(function() {
    var canvas, ctx, width, height, particles = [];
    
    function init() {
        canvas = document.createElement("canvas");
        canvas.style.cssText = "position:fixed;top:0;left:0;pointer-events:none;z-index:999999";
        document.body.appendChild(canvas);
        ctx = canvas.getContext("2d");
        resize();
        window.addEventListener("resize", resize);
        document.addEventListener("mousedown", renderFireworks);
        loop();
    }

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    function Particle(x, y) {
        this.x = x;
        this.y = y;
        
        // 1. 颜色处理：分离填充色和边框色
        var hue = Math.random() * 360;
        // 填充色：HSLA，最后一位 0.4 表示 40% 不透明度（通透感）
        this.fillColor = "hsla(" + hue + ", 100%, 75%, 0.4)";
        // 边框色：HSL，不透明，保持清晰
        this.borderColor = "hsl(" + hue + ", 100%, 75%)";
        
        // 2. 初始大小
        this.size = Math.random() * 15 + 10;
        
        // 3. 初始速度
        var angle = Math.random() * Math.PI * 2;
        var speed = Math.random() * 5 + 2;
        this.speedX = Math.cos(angle) * speed;
        this.speedY = Math.sin(angle) * speed;
    }

    Particle.prototype.update = function() {
        this.x += this.speedX;
        this.y += this.speedY;
        
        // 摩擦力：让气泡迅速停在原地
        this.speedX *= 0.92;
        this.speedY *= 0.92;
        
        // 缩小体积
        this.size -= 0.3; 
    };

    Particle.prototype.draw = function() {
        if (this.size <= 0) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        
        // --- 核心修改：先填充，后描边 ---
        
        // 1. 填充半透明内部
        ctx.fillStyle = this.fillColor;
        ctx.fill();
        
        // 2. 绘制清晰边框
        ctx.strokeStyle = this.borderColor;
        ctx.lineWidth = 2; // 边框宽度，2px 比较清晰
        ctx.stroke();
    };

    function renderFireworks(e) {
        for (var i = 0; i < 20; i++) {
            particles.push(new Particle(e.clientX, e.clientY));
        }
    }

    function loop() {
        requestAnimationFrame(loop);
        ctx.clearRect(0, 0, width, height);
        
        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            p.update();
            p.draw();
            if (p.size <= 0.1) {
                particles.splice(i, 1);
                i--;
            }
        }
    }

    init();
})();