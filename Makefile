CXX = gcc-13
CXXFLAGS = -fopenmp -g -Wno-long-long -I -L/usr/include -lgsl -lgslcblas -Wall -lm -O3 -pedantic -std=c++17
OBJS = lenski_sim.o
EXECS = lenski_main lenski_vary_epi lenski_vary_clonal

all: 
	@echo Making executables: $(EXECS)
	$(MAKE) executables

executables: $(OBJS) $(EXECS)

%.o: %.cc %.hh
	@echo Making $@ ...
	$(CXX) -c -o $@ $< $(CXXFLAGS)

lenski_main: lenski_main.cc $(OBJS)
	@echo Making $@ ...
	$(CXX) -o $@ $^ $(CXXFLAGS)

lenski_vary_epi: lenski_vary_epi.cc $(OBJS)
	@echo Making $@ ...
	$(CXX) -o $@ $^ $(CXXFLAGS)

lenski_vary_clonal: lenski_vary_clonal.cc $(OBJS)
	@echo Making $@ ...
	$(CXX) -o $@ $^ $(CXXFLAGS)

.PHONY: clean 

clean: 
	rm *.o
	rm $(EXECS)
