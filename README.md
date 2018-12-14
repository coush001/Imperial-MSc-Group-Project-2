# Smoothed Particle Hydrodynamics Python Package
Applying Computational Science and Engineering (ACSE-4): Team Arctic

This project is a basic SPH simulator. Written in python 3. The package has many helpful user features for running the simulator: Command line programme, Data file outputs, MP4 movie generation.

## Getting Started

To get up and running with our software please follow the instillation instructions:

- Dowload / Clone the repository onto your computer
- Install the required python package dependancies with the following command:
```
$pip install -r requirements.txt
```

## Usage

There are two levels of usage we envision from users

1. The developer user
Someone who has experience with python and the field of SPH who will be able to take our code and build on top of it. For these people we encourage reading through the code and understanding how it works but to get you started the following lines could be used in a python script to initialise and run your first simulation:
```
import sph_stub as sphClass

domain = sphClass.SPH_main() # Creating an instance of the class

# Set parameters for the simulation
domain.set_values(min_x=(0.0, 0.0), max_x=(10, 7), dx=0.5, h_fac=1.3, t0=0.0, t_max=3, dt=0, C_CFL=0.2, stencil=True)) 

# Set up grid initialisation
domain.initialise_grid()

# Manually place points in the domain, currently set as the classical dam break experiment
domain.place_points()

# Place particles in appropriote bucket
domain.allocate_to_grid()

# This is the function that will run the simulation for you
domain.simulate(scheme=domain.forward_euler, n=10)
```
2. User only
For people without knowledge of either python or smoothed particle hydrodynamics we have constructed a very simple to use command line programme conveniently named 'CommandLine.py' . Help docs for this programme can be found by running the following:

```
$ python CommandLine.py -h


usage: CommandLine.py [-h] [-x XDOMAIN [XDOMAIN ...]]
                      [-y YDOMAIN [YDOMAIN ...]] [-m] [-f FRAMES] [-s {fe,pc}]
                      t_max dx

positional arguments:
  t_max                 Simulation end time
  dx                    Initial Particle Spacing

optional arguments:
  -h, --help            show this help message and exit
  -x XDOMAIN [XDOMAIN ...], --xdomain XDOMAIN [XDOMAIN ...]
                        X Fluid Domain range, input: xmin xmax
  -y YDOMAIN [YDOMAIN ...], --ydomain YDOMAIN [YDOMAIN ...]
                        Y Fluid Domain range, input: ymin ymax
  -m, --movie           Save to MP4, or just show?
  -f FRAMES, --frames FRAMES
                        Save data every nth frame
  -s {fe,pc}, --scheme {fe,pc}
                        Time step scheme, choose 'fe' for forward euler or
                        'pc' for predictor corrector
```
#### Postional arguments: 
  - t_max ( Simulation length )
  - dx ( Initial partical spacing )
#### Optional arguments: 
  - -x xdomain (range of x) example= '-x 0 20'
  - -y ydomain (range of y) example= '-y 0 20'
  - -m movie (store movie flag) example '-m'
  - -f frames (frequency to store frames to animation) example= '-f'
  - -s scheme (time stepping scheme, options fe or pc for forward euler or predictor corrector) example= '-s fe'
  
An example usage of this would be:
```
$ python CommandLine.py 30 0.2 -x 0 20 -y 0 10 -m -f 10 -s pc
```

## Outputs:


Explain what output files the user can expect from using the programme, what format is the csv in etc...

How would a user change what out put they get


## Built With

Python 3

## Contributing

Please contact a team member directly if you would like to be involved with the development of this software.

## Versioning

?? 

## Authors

* **Adanna**
* **Hugo**
* **Laura**
* **Leo**
* **Wade**


See also the list of [contributors](https://github.com/msc-acse/acse-4-project-2-arctic/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thankyou to Stephen Neethling for the intial skeleton code that kickstarted this project
* Thankyou to family and friends of the developers for the support during the project
