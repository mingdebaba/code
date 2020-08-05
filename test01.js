//1,将JS嵌入元素中
//内容排版引擎处理
<html>
    <body>
        <button onclick="console.log('Hello World');">
                打印消息
        </button>
    </body>
</html>//脚本解释引擎处理
//2，将JS代码嵌入在<script>标记中，可在任何位置
<html>
    <body>
        页头
        <hr/>
        <script>
                document.write('...');
                console.log('...');
        </script>
        <hr/>
        页尾
    </body>
</html>
//3，JS代码写在外部脚本
<html>
    <head>
        <script src="myscript.js"></script>
    </head>
    <body>
    </body>
</html>
//js的添加方式 1行内，2内嵌，3外部
//行内添加就是将JS写在某一个标签之内。
<button onclick="alert('行内JS')">单击</button>
//内嵌写在head里面
<script>
    alert(‘...’);
</script>
 //外部，新建一个后缀为js的文件，然后再其他文件中引用，
// 文件中添加script标签，然后在设置src属性，
 <script>
 <script src="pop.js">
 </script>