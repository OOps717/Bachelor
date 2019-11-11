function myFunction() {
	var num;
	num=document.getElementById("myNumber").value;
	var twohund,onehund,fifty,twenty,ten,five,one;
	var k=num;
	var a=Math.floor(k/100)*100;
	var b=Math.floor((k-a)/10)*10;
	var c=k-a-b;
	document.getElementById("demo").innerHTML +=" " + num;
	twohund=Math.floor(a/200);
	document.getElementById("demo1").innerHTML +=" " + twohund;
	onehund=(a-twohund*200)/100;
	document.getElementById("demo2").innerHTML +=" " + onehund;
	fifty=(Math.floor(b/50));
	document.getElementById("demo3").innerHTML +=" " + fifty;
	twenty=Math.floor((b-fifty*50)/20);
	document.getElementById("demo4").innerHTML +=" " + twenty;
	ten=b-fifty*50-twenty*20;
	document.getElementById("demo5").innerHTML +=" " + ten;
	five=Math.floor(c/5);
	document.getElementById("demo6").innerHTML +=" " + five;
	one=c-five*5;
	document.getElementById("demo7").innerHTML +=" " + one;
}