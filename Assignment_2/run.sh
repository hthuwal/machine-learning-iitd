if [ $# -ne 4 ] then
	echo "4 Arguments Expected"
	exit 1
fi

if [ $1 -eq 1 ] then
	if [ $2 -eq 1 ] || [ $2 -eq 2 ] || [ $2 -eq 3 ] then
		cd Q1
		if [ -f $3 ] then
			python "naive_bayes_ta.py" $2 $3 $4
		else
			echo "The path $3 does not exist"
		fi
		cd ..
	else
		echo "The second argument should either 1, 2 or 3"
	fi
elif [ $1 -eq 2 ] then 
	if [ $2 -eq 1 ] then
		cd Q2
		if [ -f $3 ] then
			python "b_one_vs_one.py" $3 $4
		else
			echo "The path $3 does not exist"
		fi
		cd ..
	else
		echo "The second argument should either 1, 2 or 3"
	fi
else
	echo "The first argument should either be 1 or 2"
fi