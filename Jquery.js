//工厂函数$()
//DOM:
    document.getElementById('myList');
//Jquery
    $("#myList");

//Jquery 增删改查

    $("...").next/Prev()  //紧邻前/后一个
    $("...").nextAll/prevAll()   //之前或之后所有元素
    $("...").siblings()   //除自己之外的所有兄弟

    //访问元素属性
    $("...").attr("属性名");  //访问
    $("...").attr("属性名",值);  //修改

    //文本操作
    text()  //读取或修改节点文本内容

    //样式，修改CSS属性，
    $("...").css("CSS属性名")      //读取CSS样式
    $("...").css("css属性名",值)  //修改CSS属性

    //批量修改class样式
    $("...").hasClass("类名")      //判断是否包含指定CLASS
    $("...").addClass("类名")       //添加class
    $("...").removeClass("类名")    //移除class

    //添加
    var $new = $("html代码片段")    //创建新元素
    $(parent).append($newelem)      //新元素结尾添加到DOM树中

//JSquery事件绑定
    $("...").bind("事件类型",function(e){})