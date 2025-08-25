fact_fun() {
	ans=1
	for i in $(seq 1 $1)
	do 
		ans=$(($ans * $i))
	done
	echo " Factorial is $ans ">>output.txt
	echo " Factorial is $ans "
}

echo " Enter value of num1: "
read num1
echo " Enter value of num2: "
read num2
sum=$(($num1 + $num2))
echo " Sum of two numbers is $sum ">>output.txt
echo " Sum of two numbers is $sum "
fact_fun $sum
difference=$(diff "week3.sh" "week3_rev1.sh")
echo " Difference between the files: $difference">>output.txt

