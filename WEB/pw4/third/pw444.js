function myFunction() {
	 var str=document.getElementById("myNumber").value;
	// var str="79927398711";
	var sum;	
	var j=3;
	var num;
    var y;


	sum=0;
	for(let i=str.length-1;i>=0;i--){
		num=parseInt(str.charAt(i));
		document.getElementById("demo").innerHTML += num+" ";
		if(j%2==0){
			num=num*2;
			if(num>9){
				y=Math.floor(num/10);
				num=num-y*10;
				num=num+y;
			}
		}
		sum+=num;
		j++;
	}

	document.getElementById("demo").innerHTML += sum+" ";
	if(sum%10==0)
	document.getElementById("demo").innerHTML += "Valid";
	else{
	document.getElementById("demo").innerHTML += "Not Valid";
	}

	//document.getElementById("demo").innerHTML += k;

	//document.getElementById("demo").innerHTML += sum;
}
