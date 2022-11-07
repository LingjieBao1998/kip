window.onload = function () {
    var lists = document.getElementsByClassName('list')[0].children;
    var wrapper = document.getElementsByClassName('wrapper')[0];
    var flag = 0;
    var load;
    for (var i = 0; i < 3; i++) {
        lists[i].index = i
        lists[i].onclick = function () {
            if (flag == 1) {
                load.hide();
                flag = 0;
            }
            load = new Loading({
                type: this.index + 1,
                tipLabel: this.innerHTML,
                wrap: wrapper
            });
            load.init();
            flag = 1;
        }
    }
    lists[3].onclick = function () {
        load.hide();
        flag = 0;
    }
}