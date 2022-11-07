/**
 * type: loading 鐨勭被鍨嬶紝榛樿1
 * tipLabel: loading 鍐呯殑鏂囨湰锛岄粯璁� loading...
 * wrap: loading 鐨勭埗绾�
 * 
 * @param {*} config 浼犲叆瀵硅薄锛堝惈type/tipLabel/wrap锛�
 */
 function Loading(config) {
    this.type = config.type || 1;
    this.tipLabel = config.tipLabel || "loading...";
    this.wrap = config.wrap || document.body;
    this.loadingWrapper = null;
}

/* 鍒濆鍖� loading 鏁堟灉锛屽湪鍘熷瀷閾句笂娣诲姞 init 鏂规硶 */
Loading.prototype.init = function () {
    this.createDom();
}

/* 鍒涘缓 loading 缁撴瀯 */
Loading.prototype.createDom = function () {
    // loading wrap鐨勫瓙鐩掑瓙锛屽嵆鏁翠釜loading鐨勫唴瀹圭洅瀛�
    var loadingWrapper = document.createElement('div');
    loadingWrapper.className = 'loading-wrapper';
    // loading type瀵瑰簲鐨勪笉鍚岀殑鍔ㄧ敾
    var loadingView = document.createElement('div');
    loadingView.className = 'loading-view';
    // loading 鍐呯殑鏂囨湰鏍囩
    var tipView = document.createElement('div');
    tipView.className = 'tip-view';
    tipView.innerText = this.tipLabel;
    // 瀵� loading type鐨勪笁绉嶆儏褰㈣繘琛屽垽鏂�
    switch (this.type) {
        case 1:
            html = `
                <div class="container1">
                    <div class="circle circle1"></div>
                    <div class="circle circle2"></div>
                    <div class="circle circle3"></div>
                    <div class="circle circle4"></div>
                </div>
                <div class="container2">
                    <div class="circle circle1"></div>
                    <div class="circle circle2"></div>
                    <div class="circle circle3"></div>
                    <div class="circle circle4"></div>
                </div>
            `;
            loadingView.innerHTML = html;
            break;
        case 2:
            var html = `
                <div class="bounce-view">
                    <div class="bounce bounce1"></div>
                    <div class="bounce bounce2"></div>
                    <div class="bounce bounce3"></div>
                </div>
           `;
            loadingView.innerHTML = html;
            break;
        case 3:
            var html = `
                <div class="wave">
                    <div class="react react1"></div>
                    <div class="react react2"></div>
                    <div class="react react3"></div>
                    <div class="react react4"></div>
                    <div class="react react5"></div>
                </div>
           `;
            loadingView.innerHTML = html;
            break;
        default:
            break;
    }
    loadingWrapper.appendChild(loadingView);
    loadingWrapper.appendChild(tipView);
    this.wrap.appendChild(loadingWrapper);
    this.loadingWrapper = loadingWrapper;
}

// 瀵筶oading闅愯棌
Loading.prototype.hide = function () {
    this.wrap.removeChild(this.loadingWrapper);
}