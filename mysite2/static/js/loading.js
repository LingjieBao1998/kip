/**
 * @param {*} config 
 */
function Loading(config) {
    this.type = config.type || 1;
    this.tipLabel = config.tipLabel || "loading...";
    this.wrap = config.wrap || document.body;
    this.loadingWrapper = null;
}

Loading.prototype.init = function () {
    this.createDom();
}

Loading.prototype.createDom = function () {
    var loadingWrapper = document.createElement('div');
    loadingWrapper.className = 'loading-wrapper';
    var loadingView = document.createElement('div');
    loadingView.className = 'loading-view';
    var tipView = document.createElement('div');
    tipView.className = 'tip-view';
    tipView.innerText = this.tipLabel;
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

Loading.prototype.hide = function () {
    this.wrap.removeChild(this.loadingWrapper);
}