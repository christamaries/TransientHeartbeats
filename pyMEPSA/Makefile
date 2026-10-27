CC=gcc
CFLAGS = -Wall -Wextra -Wno-unused-variable -fpic # -Werror

all: mepsa.so

mepsa.so: mepsa.o
	$(CC) -shared -o mepsa.so mepsa.o

mepsa.o: mepsa.c
	$(CC) $(CFLAGS) -c mepsa.c	

clean:
	rm -f mepsa.o mepsa.so
