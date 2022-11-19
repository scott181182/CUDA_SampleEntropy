


build: target/sampen

target/sampen: src/main.cu
	nvcc --output-file target/sampen src/main.cu 

run: target/sampen
	./target/sampen
	