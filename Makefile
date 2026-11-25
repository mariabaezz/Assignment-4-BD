CC = mpicc
CFLAGS = -O3 -std=c11 -Wall -Wextra
TARGET = mpi_matmul

all: $(TARGET)

$(TARGET): mpi_matmul.c
	$(CC) $(CFLAGS) mpi_matmul.c -o $(TARGET)

clean:
	rm -f $(TARGET)
