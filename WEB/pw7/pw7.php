 <!DOCTYPE html>
<html lang=en>
<head>
	<meta charset="UTF-8">
	<meta name="viewpoint" content="width=device-width, initial-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">
	<title>Checker</title>
</head>
<body>

<?php
	echo "Hi,";
	echo $_PORT['name']; ?>
<br>
<?php
	echo "\nYour answers:";
	echo "<pre>";
	print_r($_POST);
	echo "<pre>"
?>

</body>
</html> 