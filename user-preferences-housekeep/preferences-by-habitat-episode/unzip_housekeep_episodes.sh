folders=(
  "/home/kartik/Documents/datasets/housekeep-episodes/train"
  "/home/kartik/Documents/datasets/housekeep-episodes/test"
  "/home/kartik/Documents/datasets/housekeep-episodes/val"
)

# Loop through the list of folders
for folder in "${folders[@]}"; do
    cd $folder
    echo $(pwd)

    for file in *.json.gz; do
        # Check if the file exists
        if [ -e "$file" ]; then
            # Unzip the file using gzip
            gzip -d "$file"
            echo "Unzipped: $file"
        else
            echo "File not found: $file"
        fi
        done
    done
done