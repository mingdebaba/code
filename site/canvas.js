var canvas,pen;
canvas = document.getElementById('myCanvas');
pen = canvas.getContext('2d');

pen.lineWidth = 1;
pen.strokeStyle = "blue";

var mousePress = false;
var last = null;

function pos(event){
    var ex,ey;
    ex = event.clientX;
    ey = event.clientY;
    return{
        x:ex,
        y:ey
    }
}


function start(event){
    mousePress = true;
}

function draw(event){
    if(!mousePress)return;
    var xy =pis(event);
    if(last != null){
        pen.beginPath();
        pen.moveTo(last.x,last.y);
        pen.lineTo(xy.x,xy.y);
        pen.strokeStyle();
    }
    last = xy;
}

function finish(event){
    mousePress = false;
    last = null;
}
canvas.onmousedown = start;
canvas.onmousemove = draw;
canvas.onmouseup = finish;