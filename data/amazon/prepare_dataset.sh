dir="dataset"

metadata="metadata.pkl"
if [ ! -f "$metadata" ]; then
	cd utils
	python parse_metadata.py
	cd ..
fi

if [[ ! -e "$dir" ]]; then
    cd utils; python create_dataset.py; cd ..
fi