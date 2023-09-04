echo $1

echo "$2"

all_args=("$@")
ds_names=("${all_args[@]:2}")

echo ds_names

echo "${ds_names[@]}"

echo "$ds_names"
