if [ $# -ne 4 ] 
then
	echo "4 Arguments Expected";
	exit 1;
fi

if [ $1 -eq 1 ] 
	then
	if [ $2 -eq 1 ] || [ $2 -eq 2 ] || [ $2 -eq 3 ] 
		then
		if [ -f $3 ] 
			then
			python "naive_bayes_ta.py" $2 $3 $4
		else
			echo "The path $3 does not exist"
		fi
	else
		echo "The second argument should either 1, 2 or 3"
	fi
elif [ $1 -eq 2 ] 
	then 
	if [ $2 -eq 1 ] 
		then
		if [ -f $3 ] 
			then
			python "one_vs_one.py" $3 $4
		else
			echo "The path $3 does not exist"
		fi
	elif [ $2 -eq 2 ] 
		then
		if [ -f $3 ] 
			then
			python "format_data_as_per_libsvm.py" $3 "temp_libsvm.csv"
			echo "Scaling the input"
			./svm-scale -l 0 -u 1 "temp_libsvm.csv" > "temp2_libsvm.csv"
			echo "Predicting"
			./svm-predict "temp2_libsvm.csv" "models/libsvm_linear_scaled.model" $4
			echo "Cleaning up..."
			rm "temp_libsvm.csv"
			rm "temp2_libsvm.csv"
		else
			echo "The path $3 does not exist"
		fi	
	fi
else
	echo "The first argument should either be 1 or 2"
fi