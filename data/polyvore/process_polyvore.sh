extract_features(){
	phase=$1
	file="../dataset/imgs_featdict_$phase.pkl"
	if [ ! -f "$file" ]; then
		python extract_features.py --phase "$phase"
	fi
}

create_dataset(){
	phase=$1
	file="../dataset/adj_$phase.npz"
	if [ ! -f "$file" ]; then
		python create_dataset.py --phase "$phase"
	fi
}

# extract img features
cd utils
extract_features "valid"
extract_features "train"
extract_features "test"

# generate adj and feature matrices
create_dataset "train"
create_dataset "valid"
create_dataset "test"
