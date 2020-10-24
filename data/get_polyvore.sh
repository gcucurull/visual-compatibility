dir="polyvore"
file="$dir/polyvore.tar.gz"
if [ ! -f "$file" ]; then
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B4Eo9mft9jwoWHhyc05VQnNyMHc' -O "$file"
fi

file_images="$dir/polyvore_images.tar.gz"
if [ ! -f "$file_images" ]; then
	gdown 'https://drive.google.com/uc?id=0B4Eo9mft9jwoNm5WR3ltVkJWX0k' -O "$file_images"
fi

if [[ ! -e "$dir/jsons" ]] || [ -z "$(ls -A $dir/jsons)" ]; then
    mkdir -p "$dir/jsons"
	tar zxfv "$file" --directory "$dir/jsons"
fi

if [[ ! -e "$dir/images" ]]; then
	tar xzvf "$file_images" --directory "$dir"
fi
