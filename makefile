CC = mpic++
CFLAGS = -Iinclude -framework Accelerate -O3 -Dp_N=$(N)

oPath = ./obj
sPath = ./src
ipath = ./include
links += -L./lib  -lparmetis -lmetis

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.h) 
sources = $(wildcard $(sPath)/*.cpp)

tempsrc := $(sources)
sources = $(filter-out ./src/MaxwellsRun3d.cpp ./src/MaxwellsRun3d_optimisedVK.cpp ./src/MaxwellsRun3d_optimisedSK.cpp ./src/MaxwellsRun3d_optimised2K.cpp ./src/MaxwellsRun3d_optimised2K_restruct.cpp, $(tempsrc))
ifeq ($(optimise), vk)
sources += ./src/MaxwellsRun3d_optimisedVK.cpp
else ifeq ($(optimise), sk)
sources += ./src/MaxwellsRun3d_optimisedSK.cpp
else ifeq ($(optimise), 2k)
sources += ./src/MaxwellsRun3d_optimised2K.cpp
else ifeq ($(optimise), 2k+)
sources += ./src/MaxwellsRun3d_optimised2K_restruct.cpp
else
sources += ./src/MaxwellsRun3d.cpp
endif

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o)) 

main: $(objects) $(headers) main.cpp
	$(CC) $(CFLAGS) -o main $(objects) main.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(CC) $(CFLAGS) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main;
#=================================================
