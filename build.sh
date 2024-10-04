#!/bin/bash
output_dir=build

mkdir -p $output_dir

cmake -S . -B $output_dir -DCMAKE_BUILD_TYPE=Debug
if [ $? != 0 ]; then
    echo ">>> Cmake error $? couldn't build the project"
    exit $?
fi

echo ">>> Building..."
cd $output_dir
make
if [ $? == 0 ]; then
    echo ">>> Done."
else
    echo ">>> Make error $?"
fi
cd -
