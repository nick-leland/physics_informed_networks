# The goal here is to fundamentally describe the heat transfer from an object to another object.

# Start by establishing a library of Thermal Diffusivity 

thermal_diffusivity_library = {
        "diamond" : (1060+1160)//2,
        "helium" : 190,
        "silver" : 165.63
        }

if __name__ == "__main__":
    print(thermal_diffusivity_library["silver"])
