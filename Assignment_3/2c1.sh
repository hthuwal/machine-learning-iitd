echo "Training"
python format_data_as_per_libsvm.py $1 temp.csv
./svm-train -s 0 -c 1 -t 0  temp.csv svm.model
echo "Accuracy on train set"
./svm-predict temp.csv svm.model out.txt
echo "Accuracy on test set"
python format_data_as_per_libsvm.py $2 temp.csv
./svm-predict temp.csv svm.model out.txt