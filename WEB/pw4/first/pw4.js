function myFunction() {
	var num,a,b,c;
	num=document.getElementById("myNumber").value;
	c=num;
	b=num;
	a=num;
	c=(c%100)%10;
	b=((b%100)-c)/10;
	a=(a-b*10-c)/100;
	document.getElementById("demo1").innerHTML = a;
	document.getElementById("demo2").innerHTML = b;
	document.getElementById("demo3").innerHTML = c;
	if(num==(Math.pow(a,3)+Math.pow(b,3)+Math.pow(c,3))){
	document.getElementById("demo").innerHTML = num;
	}else{
		 for(let a=1;a<=9;a++){
		 	for(let b=1;b<=9;b++){
		 		for(let c=1;c<=9;c++){
		 			if((Math.pow(a,3)+Math.pow(b,3)+Math.pow(c,3)) == a * 100 + b * 10 + c){
		 				document.getElementById("demo").innerHTML += a*100+b*10+c + " ";
		 				console.log(a * 100 + b * 10 + c);
		 			}
		 		}
		 	}
		 }
	}
}