galaxies: galaxies.cpp
	g++ -O2 -std=c++17 -Wall -Wno-unused-result galaxies.cpp -o galaxies -lglfw -lepoxy -lpulse-simple -lpulse -pthread

clean:
	rm -f galaxies
