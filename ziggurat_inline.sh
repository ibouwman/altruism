#! /bin/bash
#
cp ziggurat_inline.h /$HOME/include
#
gcc -c -Wall ziggurat_inline.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
mv ziggurat_inline.o ~/libc/ziggurat_inline.o
#
echo "Normal end of execution."
