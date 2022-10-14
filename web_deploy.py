#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import dataset


#load the model from disk
import joblib
filename = 'finalized_model_.sav'
model = joblib.load(filename)

#Import python scripts
#from preprocessing import preprocess


def main(): 
    
    data={
        'Unnamed: 0' : 0,
        'fuel_type_Diesel': 0, 
        'fuel_type_Electric': 0, 
        'fuel_type_Electric/Diesel': 0, 
        'fuel_type_Electric/Gasoline': 0, 
        'fuel_type_Gasoline': 0,
        'fuel_type_LPG': 0, 
        'fuel_type_Others': 0, 
        'seller_Dealer': 0, 
        'seller_Privateseller': 0, 
        'body_type_Compact': 0, 
        'body_type_Convertible': 0, 
        'body_type_Coupe': 0, 
        'body_type_Off-Road/Pick-up': 0, 
        'body_type_Other': 0, 
        'body_type_Sedan': 0, 
        'body_type_Station wagon': 0, 
        'body_type_Transporter': 0, 
        'body_type_Van': 0, 
        'type_New': 0, 
        'type_Used': 0, 
        'drivetrain_4WD': 0, 
        'drivetrain_Front': 0, 
        'drivetrain_Rear': 0, 
        'Gearbox_Automatic': 0, 
        'Gearbox_Manual': 0, 
        'Gearbox_Semi-automatic': 0, 
        'gears_1': 0, 
        'gears_2': 0, 
        'gears_3': 0, 
        'gears_4': 0, 
        'gears_5': 0, 
        'gears_6': 0, 
        'gears_7': 0, 
        'gears_8': 0, 
        'gears_9': 0, 
        'colour_Beige': 0, 
        'colour_Black': 0, 
        'colour_Blue': 0, 
        'colour_Bronze': 0, 
        'colour_Brown': 0, 
        'colour_Gold': 0, 
        'colour_Green': 0, 
        'colour_Grey': 0, 
        'colour_Orange': 0, 
        'colour_Red': 0, 
        'colour_Silver': 0, 
        'colour_Violet': 0, 
        'colour_White': 0, 
        'colour_Yellow': 0, 
        'upholstery_colour_Beige': 0, 
        'upholstery_colour_Black': 0, 
        'upholstery_colour_Blue': 0, 
        'upholstery_colour_Brown': 0, 
        'upholstery_colour_Green': 0, 
        'upholstery_colour_Grey': 0, 
        'upholstery_colour_Orange': 0,
        'upholstery_colour_Other': 0, 
        'upholstery_colour_Red': 0, 
        'upholstery_colour_White': 0,
        'upholstery_colour_Yellow': 0, 
        'upholstery_Cloth': 0, 
        'upholstery_Full leather': 0, 
        'upholstery_Other': 0, 
        'upholstery_Part leather': 0, 
        'upholstery_Velour': 0,
        'upholstery_alcantara': 0, 
        'make_model_Audi-A1': 0, 
        'make_model_Audi-A3': 0, 
        'make_model_Audi-A4': 0, 
        'make_model_Audi-A4 allroad': 0, 
        'make_model_Audi-A5': 0, 
        'make_model_Audi-A6': 0, 
        'make_model_Audi-A6 allroad': 0, 
        'make_model_Audi-A7': 0, 
        'make_model_Audi-A8': 0, 
        'make_model_Audi-Q2': 0, 
        'make_model_Audi-Q3': 0, 
        'make_model_Audi-Q4 e-tron': 0, 
        'make_model_Audi-Q5': 0, 
        'make_model_Audi-Q7': 0, 
        'make_model_Audi-Q8': 0, 
        'make_model_Audi-R8': 0, 
        'make_model_Audi-S5': 0, 
        'make_model_Audi-SQ5': 0, 
        'make_model_Audi-SQ7': 0, 
        'make_model_Audi-TT': 0, 
        'make_model_Audi-e-tron': 0, 
        'make_model_Audi-e-tron GT': 0, 
        'make_model_BMW-116': 0, 
        'make_model_BMW-118': 0, 
        'make_model_BMW-120': 0, 
        'make_model_BMW-216': 0, 
        'make_model_BMW-218': 0, 
        'make_model_BMW-220': 0, 
        'make_model_BMW-225': 0, 
        'make_model_BMW-316': 0, 
        'make_model_BMW-318': 0, 
        'make_model_BMW-320': 0, 
        'make_model_BMW-325': 0, 
        'make_model_BMW-328': 0, 
        'make_model_BMW-330': 0, 
        'make_model_BMW-335': 0, 
        'make_model_BMW-418': 0, 
        'make_model_BMW-420': 0, 
        'make_model_BMW-428': 0, 
        'make_model_BMW-435': 0, 
        'make_model_BMW-520': 0, 
        'make_model_BMW-523': 0, 
        'make_model_BMW-525': 0, 
        'make_model_BMW-530': 0, 
        'make_model_BMW-535': 0, 
        'make_model_BMW-640': 0, 
        'make_model_BMW-650': 0, 
        'make_model_BMW-730': 0, 
        'make_model_BMW-740': 0, 
        'make_model_BMW-M3': 0, 
        'make_model_BMW-M4': 0, 
        'make_model_BMW-Others': 0, 
        'make_model_BMW-X1': 0, 
        'make_model_BMW-X2': 0, 
        'make_model_BMW-X3': 0, 
        'make_model_BMW-X5': 0, 
        'make_model_BMW-X6': 0, 
        'make_model_BMW-Z3': 0, 
        'make_model_BMW-Z4': 0, 
        'make_model_BMW-i3': 0, 
        'make_model_BMW-i4': 0, 
        'make_model_BMW-iX': 0, 
        'make_model_BMW-iX3': 0, 
        'make_model_Chevrolet-Aveo': 0, 
        'make_model_Chevrolet-Camaro': 0, 
        'make_model_Chevrolet-Captiva': 0, 
        'make_model_Chevrolet-Corvette': 0, 
        'make_model_Chevrolet-Cruze': 0, 
        'make_model_Chevrolet-Spark': 0, 
        'make_model_Citroen-Berlingo': 0, 
        'make_model_Citroen-C1': 0, 
        'make_model_Citroen-C3': 0, 
        'make_model_Citroen-C3 Aircross': 0, 
        'make_model_Citroen-C3 Picasso': 0, 
        'make_model_Citroen-C4': 0, 
        'make_model_Citroen-C4 Cactus': 0, 
        'make_model_Citroen-C4 Picasso': 0, 
        'make_model_Citroen-C5': 0, 
        'make_model_Citroen-C5 Aircross': 0,
        'make_model_Citroen-DS3': 0, 
        'make_model_Citroen-Grand C4 Picasso': 0, 
        'make_model_Citroen-Jumper': 0, 
        'make_model_Citroen-Jumpy': 0, 
        'make_model_Citroen-Others': 0, 
        'make_model_Citroen-Xsara Picasso': 0, 
        'make_model_Dacia-Dokker': 0, 
        'make_model_Dacia-Duster': 0, 
        'make_model_Dacia-Lodgy': 0, 
        'make_model_Dacia-Logan': 0, 
        'make_model_Dacia-Sandero': 0, 
        'make_model_Dacia-Spring': 0, 
        'make_model_Fiat-500': 0, 
        'make_model_Fiat-500C': 0, 
        'make_model_Fiat-500L': 0, 
        'make_model_Fiat-500X': 0, 
        'make_model_Fiat-500e': 0, 
        'make_model_Fiat-Doblo': 0,
        'make_model_Fiat-Ducato': 0, 
        'make_model_Fiat-Grande Punto': 0, 
        'make_model_Fiat-Panda': 0, 
        'make_model_Fiat-Punto': 0, 
        'make_model_Fiat-Punto Evo': 0, 
        'make_model_Fiat-Stilo': 0, 
        'make_model_Fiat-Talento': 0, 
        'make_model_Fiat-Tipo': 0, 
        'make_model_Ford-B-Max': 0, 
        'make_model_Ford-C-Max': 0, 
        'make_model_Ford-EcoSport': 0, 
        'make_model_Ford-Explorer': 0, 
        'make_model_Ford-F 150': 0, 
        'make_model_Ford-Fiesta': 0, 
        'make_model_Ford-Focus': 0, 
        'make_model_Ford-Focus C-Max': 0, 
        'make_model_Ford-Fusion': 0, 
        'make_model_Ford-Galaxy': 0, 
        'make_model_Ford-Ka/Ka+': 0, 
        'make_model_Ford-Kuga': 0, 
        'make_model_Ford-Mondeo': 0, 
        'make_model_Ford-Mustang': 0, 
        'make_model_Ford-Mustang Mach-E': 0, 
        'make_model_Ford-Puma': 0, 
        'make_model_Ford-Ranger': 0, 
        'make_model_Ford-S-Max': 0, 
        'make_model_Ford-Transit': 0, 
        'make_model_Ford-Transit Connect': 0, 
        'make_model_Ford-Transit Custom': 0, 
        'make_model_Honda-Accord': 0, 
        'make_model_Honda-CR-V': 0, 
        'make_model_Honda-Civic': 0, 
        'make_model_Honda-HR-V': 0, 
        'make_model_Honda-Insight': 0, 
        'make_model_Honda-Jazz': 0, 
        'make_model_Honda-e': 0, 
        'make_model_Hyundai-Bayon': 0, 
        'make_model_Hyundai-Ioniq': 0, 
        'make_model_Hyundai-Ioniq 5': 0, 
        'make_model_Hyundai-Kona': 0, 
        'make_model_Hyundai-Nexo': 0, 
        'make_model_Hyundai-Santa Fe': 0, 
        'make_model_Hyundai-Tucson': 0, 
        'make_model_Hyundai-i10': 0, 
        'make_model_Hyundai-i20': 0, 
        'make_model_Hyundai-i30': 0, 
        'make_model_Hyundai-i40': 0, 
        'make_model_Hyundai-iX20': 0, 
        'make_model_Hyundai-iX35': 0, 
        'make_model_Kia-Carens': 0, 
        "make_model_Kia-Ceed / cee'd": 0, 
        "make_model_Kia-Ceed SW / cee'd SW": 0, 
        'make_model_Kia-EV6': 0, 
        'make_model_Kia-Niro': 0, 
        'make_model_Kia-Optima': 0, 
        'make_model_Kia-Picanto': 0,
        "make_model_Kia-ProCeed / pro_cee'd": 0, 
        'make_model_Kia-Rio': 0, 
        'make_model_Kia-Sorento': 0, 
        'make_model_Kia-Soul': 0, 
        'make_model_Kia-Sportage': 0, 
        'make_model_Kia-Stonic': 0, 
        'make_model_Kia-Venga': 0, 
        'make_model_Kia-XCeed': 0, 
        'make_model_Kia-e-Niro': 0, 
        'make_model_Mazda-2': 0, 
        'make_model_Mazda-3': 0, 
        'make_model_Mazda-5': 0, 
        'make_model_Mazda-6': 0, 
        'make_model_Mazda-CX-3': 0, 
        'make_model_Mazda-CX-30': 0, 
        'make_model_Mazda-CX-5': 0, 
        'make_model_Mazda-MX-30': 0, 
        'make_model_Mazda-MX-5': 0, 
        'make_model_Mercedes-Benz-A 150': 0, 
        'make_model_Mercedes-Benz-A 160': 0, 
        'make_model_Mercedes-Benz-A 170': 0, 
        'make_model_Mercedes-Benz-A 180': 0, 
        'make_model_Mercedes-Benz-A 200': 0, 
        'make_model_Mercedes-Benz-A 250': 0, 
        'make_model_Mercedes-Benz-A 45 AMG': 0, 
        'make_model_Mercedes-Benz-B 160': 0, 
        'make_model_Mercedes-Benz-B 170': 0, 
        'make_model_Mercedes-Benz-B 180': 0, 
        'make_model_Mercedes-Benz-B 200': 0, 
        'make_model_Mercedes-Benz-B 250': 0, 
        'make_model_Mercedes-Benz-C 180': 0, 
        'make_model_Mercedes-Benz-C 200': 0, 
        'make_model_Mercedes-Benz-C 220': 0, 
        'make_model_Mercedes-Benz-C 250': 0, 
        'make_model_Mercedes-Benz-C 300': 0, 
        'make_model_Mercedes-Benz-C 350': 0, 
        'make_model_Mercedes-Benz-CLA 180': 0, 
        'make_model_Mercedes-Benz-CLA 200': 0, 
        'make_model_Mercedes-Benz-CLA 250': 0, 
        'make_model_Mercedes-Benz-CLK 200': 0, 
        'make_model_Mercedes-Benz-CLS 350': 0, 
        'make_model_Mercedes-Benz-Citan': 0, 
        'make_model_Mercedes-Benz-E 200': 0, 
        'make_model_Mercedes-Benz-E 220': 0, 
        'make_model_Mercedes-Benz-E 240': 0, 
        'make_model_Mercedes-Benz-E 250': 0, 
        'make_model_Mercedes-Benz-E 300': 0, 
        'make_model_Mercedes-Benz-E 320': 0, 
        'make_model_Mercedes-Benz-E 350': 0, 
        'make_model_Mercedes-Benz-EQC 400': 0, 
        'make_model_Mercedes-Benz-EQS': 0, 
        'make_model_Mercedes-Benz-GLA 180': 0, 
        'make_model_Mercedes-Benz-GLA 200': 0, 
        'make_model_Mercedes-Benz-GLA 250': 0, 
        'make_model_Mercedes-Benz-GLC 220': 0, 
        'make_model_Mercedes-Benz-GLC 250': 0, 
        'make_model_Mercedes-Benz-GLC 300': 0, 
        'make_model_Mercedes-Benz-GLC 350': 0, 
        'make_model_Mercedes-Benz-GLE 350': 0, 
        'make_model_Mercedes-Benz-ML 320': 0, 
        'make_model_Mercedes-Benz-ML 350': 0, 
        'make_model_Mercedes-Benz-Others': 0, 
        'make_model_Mercedes-Benz-S 350': 0, 
        'make_model_Mercedes-Benz-SL 500': 0, 
        'make_model_Mercedes-Benz-SLK 200': 0, 
        'make_model_Mercedes-Benz-Sprinter': 0, 
        'make_model_Mercedes-Benz-Vito': 0, 
        'make_model_Opel-Adam': 0, 
        'make_model_Opel-Agila': 0, 
        'make_model_Opel-Ampera': 0, 
        'make_model_Opel-Astra': 0, 
        'make_model_Opel-Cascada': 0, 
        'make_model_Opel-Combo': 0, 
        'make_model_Opel-Corsa': 0, 
        'make_model_Opel-Corsa-e': 0, 
        'make_model_Opel-Crossland X': 0, 
        'make_model_Opel-Grandland X': 0, 
        'make_model_Opel-Insignia': 0, 
        'make_model_Opel-Karl': 0, 
        'make_model_Opel-Meriva': 0, 
        'make_model_Opel-Mokka': 0, 
        'make_model_Opel-Mokka X': 0, 
        'make_model_Opel-Mokka-E': 0, 
        'make_model_Opel-Tigra': 0, 
        'make_model_Opel-Vectra': 0, 
        'make_model_Opel-Vivaro': 0, 
        'make_model_Peugeot-108': 0, 
        'make_model_Peugeot-2008': 0, 
        'make_model_Peugeot-206': 0, 
        'make_model_Peugeot-207': 0, 
        'make_model_Peugeot-208': 0, 
        'make_model_Peugeot-3008': 0,
        'make_model_Peugeot-307': 0, 
        'make_model_Peugeot-308': 0, 
        'make_model_Peugeot-407': 0, 
        'make_model_Peugeot-5008': 0,
        'make_model_Peugeot-508': 0, 
        'make_model_Peugeot-Boxer': 0, 
        'make_model_Peugeot-Expert': 0,
        'make_model_Peugeot-Partner': 0, 
        'make_model_Peugeot-RCZ': 0, 
        'make_model_Peugeot-Rifter': 0, 
        'make_model_Renault-Arkana': 0, 
        'make_model_Renault-Captur': 0, 
        'make_model_Renault-Clio': 0, 
        'make_model_Renault-Espace': 0,
        'make_model_Renault-Grand Scenic': 0, 
        'make_model_Renault-Kadjar': 0, 
        'make_model_Renault-Kangoo': 0, 
        'make_model_Renault-Laguna': 0, 
        'make_model_Renault-Master': 0, 
        'make_model_Renault-Megane': 0, 
        'make_model_Renault-Modus': 0, 
        'make_model_Renault-Scenic': 0, 
        'make_model_Renault-Talisman': 0, 
        'make_model_Renault-Trafic': 0, 
        'make_model_Renault-Twingo': 0, 
        'make_model_Renault-ZOE': 0, 
        'make_model_Skoda-Citigo': 0, 
        'make_model_Skoda-Enyaq': 0, 
        'make_model_Skoda-Fabia': 0, 
        'make_model_Skoda-Kamiq': 0, 
        'make_model_Skoda-Karoq': 0, 
        'make_model_Skoda-Kodiaq': 0,
        'make_model_Skoda-Octavia': 0, 
        'make_model_Skoda-Rapid/Spaceback': 0,
        'make_model_Skoda-Roomster': 0, 
        'make_model_Skoda-Scala': 0, 
        'make_model_Skoda-Superb': 0,
        'make_model_Skoda-Yeti': 0, 
        'make_model_Tesla-Model 3': 0, 
        'make_model_Tesla-Model S': 0, 
        'make_model_Tesla-Model X': 0, 
        'make_model_Toyota-Auris': 0, 
        'make_model_Toyota-Avensis': 0,
        'make_model_Toyota-Aygo': 0, 
        'make_model_Toyota-C-HR': 0, 
        'make_model_Toyota-Camry': 0,
        'make_model_Toyota-Corolla': 0, 
        'make_model_Toyota-Corolla Verso': 0, 
        'make_model_Toyota-Hilux': 0, 
        'make_model_Toyota-Land Cruiser': 0,
        'make_model_Toyota-Mirai': 0, 
        'make_model_Toyota-Prius': 0, 
        'make_model_Toyota-Prius+': 0,
        'make_model_Toyota-Proace': 0,
        'make_model_Toyota-RAV 4': 0, 
        'make_model_Toyota-Verso': 0, 
        'make_model_Toyota-Verso-S': 0, 
        'make_model_Toyota-Yaris': 0, 
        'make_model_Volkswagen-Amarok': 0, 
        'make_model_Volkswagen-Arteon': 0, 
        'make_model_Volkswagen-Beetle': 0, 
        'make_model_Volkswagen-Caddy': 0,
        'make_model_Volkswagen-Crafter': 0,
        'make_model_Volkswagen-Eos': 0, 
        'make_model_Volkswagen-Golf': 0, 
        'make_model_Volkswagen-Golf Cabriolet': 0,
        'make_model_Volkswagen-Golf GTE': 0, 
        'make_model_Volkswagen-Golf GTI': 0, 
        'make_model_Volkswagen-Golf Plus': 0, 
        'make_model_Volkswagen-Golf Sportsvan': 0, 
        'make_model_Volkswagen-Golf Variant': 0, 
        'make_model_Volkswagen-ID.3': 0, 
        'make_model_Volkswagen-ID.4': 0, 
        'make_model_Volkswagen-Jetta': 0, 
        'make_model_Volkswagen-Others': 0, 
        'make_model_Volkswagen-Passat': 0, 
        'make_model_Volkswagen-Passat CC': 0, 
        'make_model_Volkswagen-Passat Variant': 0,
        'make_model_Volkswagen-Polo': 0, 
        'make_model_Volkswagen-Scirocco': 0,
        'make_model_Volkswagen-Sharan': 0, 
        'make_model_Volkswagen-T-Cross': 0, 
        'make_model_Volkswagen-T-Roc': 0, 
        'make_model_Volkswagen-T5 Transporter': 0, 
        'make_model_Volkswagen-T6 Transporter': 0, 
        'make_model_Volkswagen-Tiguan': 0,
        'make_model_Volkswagen-Touareg': 0,
        'make_model_Volkswagen-Touran': 0, 
        'make_model_Volkswagen-Transporter': 0, 
        'make_model_Volkswagen-e-Golf': 0, 
        'make_model_Volkswagen-e-up!': 0, 
        'make_model_Volkswagen-up!': 0, 
        'make_model_Volvo-C30': 0, 
        'make_model_Volvo-C70': 0, 
        'make_model_Volvo-S40': 0, 
        'make_model_Volvo-S60': 0, 
        'make_model_Volvo-S80': 0, 
        'make_model_Volvo-S90': 0, 
        'make_model_Volvo-V40': 0, 
        'make_model_Volvo-V40 Cross Country': 0,
        'make_model_Volvo-V50': 0, 
        'make_model_Volvo-V60': 0, 
        'make_model_Volvo-V60 Cross Country': 0,
        'make_model_Volvo-V70': 0, 
        'make_model_Volvo-V90': 0, 
        'make_model_Volvo-XC40': 0,
        'make_model_Volvo-XC60': 0, 
        'make_model_Volvo-XC70': 0, 
        'make_model_Volvo-XC90': 0, 
        'make_country_France': 0, 
        'make_country_Germany': 0, 
        'make_country_Italy': 0,
        'make_country_Japan': 0, 
        'make_country_Korea': 0, 
        'make_country_Sweden': 0, 
        'make_country_USA': 0, 
        'province_Drenthe': 0, 
        'province_Flevoland': 0, 
        'province_Friesland': 0, 
        'province_Gelderland': 0, 
        'province_Groningen': 0, 
        'province_Limburg': 0, 
        'province_North Brabant': 0, 
        'province_North Holland': 0, 
        'province_Overijssel': 0, 
        'province_South Holland': 0, 
        'province_Utrecht': 0, 
        'province_Zeeland': 0,  
        'mileage': 0, 
        'seats': 0, 
        'doors': 0, 
        'warranty': 0, 
        'first_registration': 0,
        'general_inspection': 0, 
        'full_service_history': 0, 
        'non_smoker_vehicle': 0, 
        'Power': 0,
        'engine_size': 0, 
        'cylinders': 0, 
        'empty_weight': 0,
        'co2_emissions': 0,
        'emission_class': 0,
        'paint': 0, 
        '_fuel_con_comb': 0, 
        'ext_All season tyres': 0,
        'ext_Alloy wheels': 0, 
        'ext_Ambient lighting': 0, 
        'ext_Automatically dimming interior mirror': 0, 
        'ext_Awning': 0, 
        'ext_Biodiesel conversion': 0, 
        'ext_Cargo barrier': 0, 
        'ext_Catalytic Converter': 0, 
        'ext_E10-enabled': 0,
        'ext_Electronic parking brake': 0,
        'ext_Emergency tyre': 0, 
        'ext_Emergency tyre repair kit': 0, 
        'ext_Handicapped enabled': 0, 
        'ext_Headlight washer system': 0, 
        'ext_Range extender': 0, 
        'ext_Right hand drive': 0, 
        'ext_Roof rack': 0, 
        'ext_Shift paddles': 0,
        'ext_Ski bag': 0, 
        'ext_Sliding door': 0, 
        'ext_Smokers package': 0,
        'ext_Spare tyre': 0,
        'ext_Spoiler': 0, 
        'ext_Sport package': 0,
        'ext_Sport seats': 0, 
        'ext_Sport suspension': 0, 
        'ext_Steel wheels': 0, 
        'ext_Summer tyres': 0, 
        'ext_Touch screen': 0, 
        'ext_Trailer hitch': 0,
        'ext_Tuned car': 0, 
        'ext_Voice Control': 0, 
        'ext_Winter package': 0,
        'ext_Winter tyres': 0,
        'age': 0, 
        'mileage_digitized': 0,
        'cc_2_zones': 0, 
        'cc_360°_camera': 0, 
        'cc_3_zones': 0, 
        'cc_4_zones': 0, 
        'cc_Air_conditioning': 0,
        'cc_Air_suspension': 0,
        'cc_Armrest': 0,
        'cc_Automatic_climate_control': 0, 
        'cc_Auxiliary_heating': 0, 
        'cc_Cruise_control': 0,
        'cc_Electric_backseat_adjustment': 0, 
        'cc_Electric_tailgate': 0, 
        'cc_Electrical_side_mirrors': 0,
        'cc_Electrically_adjustable_seats': 0,
        'cc_Electrically_heated_windshield': 0,
        'cc_Fold_flat_passenger_seat': 0, 
        'cc_Heads_up_display': 0, 
        'cc_Heated_steering_wheel': 0, 
        'cc_Hill_Holder': 0, 
        'cc_Keyless_central_door_lock': 0, 
        'cc_Leather_seats': 0, 
        'cc_Leather_steering_wheel': 0,
        'cc_Light_sensor': 0,
        'cc_Lumbar_support': 0, 
        'cc_Massage_seats': 0,
        'cc_Multi_function_steering_wheel': 0,
        'cc_Navigation_system': 0, 
        'cc_Panorama_roof': 0, 
        'cc_Park_Distance_Control': 0,
        'cc_Parking_assist_system_camera': 0,
        'cc_Parking_assist_system_self_steering': 0,
        'cc_Parking_assist_system_sensors_front': 0, 
        'cc_Parking_assist_system_sensors_rear': 0, 
        'cc_Power_windows': 0, 
        'cc_Rain_sensor': 0, 
        'cc_Seat_heating': 0, 
        'cc_Seat_ventilation': 0, 
        'cc_Sliding_door_left': 0,
        'cc_Sliding_door_right': 0, 
        'cc_Split_rear_seats': 0, 
        'cc_Start_stop_system': 0, 
        'cc_Sunroof': 0, 
        'cc_Tinted_windows': 0, 
        'cc_Wind_deflector': 0, 
        'ent_Android_Auto': 0, 
        'ent_Apple_CarPlay': 0, 
        'ent_Bluetooth': 0, 
        'ent_CD_player': 0, 
        'ent_Digital_cockpit': 0, 
        'ent_Digital_radio': 0,
        'ent_Hands-free_equipment': 0, 
        'ent_Induction_charging_for_smartphones': 0, 
        'ent_Integrated_music_streaming': 0, 
        'ent_MP3': 0, 
        'ent_On-board_computer': 0,
        'ent_Radio': 0, 
        'ent_Sound_system': 0,
        'ent_Television': 0,
        'ent_USB': 0, 
        'ent_WLAN_/_WiFi_hotspot': 0, 
        'ss_ABS': 0, 
        'ss_Adaptive_Cruise_Control': 0, 
        'ss_Adaptive_headlights': 0, 
        'ss_Alarm_system': 0, 
        'ss_Bi_Xenon_headlights': 0, 
        'ss_Blind_spot_monitor': 0,
        'ss_Central_door_lock': 0, 
        'ss_Central_door_lock_with_remote_control': 0,
        'ss_Daytime_running_lights': 0, 
        'ss_Distance_warning_system': 0, 
        'ss_Driver_drowsiness_detection': 0,
        'ss_Driver_side_airbag': 0, 
        'ss_Electronic_stability_control': 0, 
        'ss_Emergency_brake_assistant': 0, 
        'ss_Emergency_system': 0, 
        'ss_Fog_lights': 0, 
        'ss_Full_LED_headlights': 0,
        'ss_Glare_free_high_beam_headlights': 0,
        'ss_Head_airbag': 0, 
        'ss_High_beam_assist': 0, 
        'ss_Immobilizer': 0, 
        'ss_Isofix': 0, 
        'ss_LED_Daytime_Running_Lights': 0,
        'ss_LED_Headlights': 0, 
        'ss_Lane_departure_warning_system': 0, 
        'ss_Laser_headlights': 0,
        'ss_Night_view_assist': 0, 
        'ss_Passenger_side_airbag': 0,
        'ss_Power_steering': 0, 
        'ss_Rear_airbag': 0,
        'ss_Side_airbag': 0, 
        'ss_Speed_limit_control_system': 0, 
        'ss_Tire_pressure_monitoring_system': 0,
        'ss_Traction_control': 0, 
        'ss_Traffic_sign_recognition': 0,
        'ss_Xenon_headlights': 0}
 
    st.title('Fenyx GroupTyping AutoScout Model App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict Price.
    The application is functional for online prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('logo2.jpeg')
    st.sidebar.image(image,)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online",))
    
    
    if add_selectbox == "Online":
        st.info("Input data below")
        
        st.subheader("Please select brand and model:")
        
        make = st.selectbox('What is the car brand?',
                            dataset.make(), index=dataset.make(most=True))
        
        model_ = st.selectbox(f'What is the model of {make}?', dataset.model(
            make), index=dataset.model(make, most=True)) 
        
        make_model_= 'make_model_'+make+'-'+model_
        data[make_model_] = 1
        
        st.subheader("Please select information of Auto:")
        
        title_0 = st.selectbox('Clour of Auto :', ('Black','Beige','Blue','Bronze','Brown','Gold','Green','Grey','Orange','Red','Silver','Violet','White','Yellow'))
        colour='colour_'+title_0
        data[colour] = 1

        title_0 = st.selectbox('Body type of Auto :', ('Other','Convertible','Coupe','Off-Road/Pick-up','Compact','Sedan','Station wagon','Transporter','Van'))
        body_type='body_type_'+title_0
        data[body_type] = 1

        title_0 = st.selectbox('Type of Auto :', ('New','Used'))
        type='type_'+title_0
        data[type] = 1
        
        title_0 = st.selectbox('Fuel Type of Auto :', ('Diesel','Electric','Electric/Diesel','Electric/Gasoline','Gasoline','LPG','Others'))
        fuel_type='fuel_type_'+title_0
        data[fuel_type] = 1
        
        title_0 = st.selectbox('Gear Box of Auto :', ('Automatic','Manual','Semi-automatic'))
        Gear_box='Gearbox_'+title_0
        data[Gear_box] = 1

        title_0 = st.selectbox('Province of Auto :', ('Drenthe','Flevoland','Friesland','Gelderland','Groningen','Limburg','North Brabant','North Holland','Overijssel','South Holland','Utrecht','Zeeland'))
        province='province_'+title_0
        data[province] = 1
        
        mileage = st.number_input('Mileage of Auto :',min_value= 1000,max_value=400000,value=105370)
        data['mileage'] = mileage

        engine_size = st.number_input('Engine Size of Auto:',min_value=875,max_value=2885,value=1697)
        data['engine_size'] = engine_size

        clinder = st.number_input('Clinder of Auto :',min_value= 0,max_value=8,value=4)
        data['cylinders'] = clinder
        
        fuel_con_comb = st.number_input('Fuel consumption of Auto :',min_value=0, max_value=11, value=5,format="%.2f")
        data['_fuel_con_comb'] = fuel_con_comb

        door = st.number_input('Doors of Auto :',min_value= 0,max_value=8,value=5)
        data['doors'] = door
         
        age = st.slider('Age of Auto :', min_value=0, max_value=22, value=7)
        data['age'] = age

        power = st.slider('Power of Auto :', min_value=32, max_value=229, value=113)
        data['Power'] = power

        emission_class = st.slider('Emission Class of Auto :', min_value=1, max_value=6, value=5) 
        data['emission_class'] = emission_class
        
        st.sidebar.subheader("Please select extra information of Auto :")

        warranty = st.sidebar.checkbox('Warranty')

        if warranty:
            data["warranty"] = 1

        air_suspension = st.sidebar.checkbox('Air Suspension')

        if air_suspension:
            data["cc_Air_suspension"] = 1

        air_conditioning = st.sidebar.checkbox('Air Conditioning')

        if air_conditioning:
            data["cc_Air_conditioning"] = 1
            
        radio = st.sidebar.checkbox('Radio')

        if radio:
            data["ent_Radio"] = 1
            
        rain_sensor = st.sidebar.checkbox('Rain Sensor')

        if rain_sensor:
            data["cc_Rain_sensor"] = 1
        
        navigation_system = st.sidebar.checkbox('Navigation System')

        if navigation_system:
            data["cc_Navigation_system"] = 1
            
       
                
        features_df = pd.DataFrame.from_dict([data]) 
        prediction = model.predict(features_df)
        if st.button('Predict'):
    
                st.success(f"The Prediction Price of the Car is €{int(prediction)}")
                
    st.sidebar.info('This app is created to predict Price')            
    # elif add_selectbox == "Batch":
    #     st.subheader("Dataset upload")
    #     uploaded_file = st.file_uploader("Choose a file")
    #     if uploaded_file is not None:
    #         data = pd.read_csv(uploaded_file,encoding= 'utf-8')
    #         #Get overview of data
    #         st.write(data.head())
    #         st.markdown("<h3></h3>", unsafe_allow_html=True)
    #         #Preprocess inputs
    #         features_df = preprocess(data, "Batch")
    #         if st.button('Predict'):
    #            #Get batch prediction
    #             prediction = model.predict(features_df)
    #             prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                

    #             st.markdown("<h3></h3>", unsafe_allow_html=True)
    #             st.subheader('Prediction')
    #             st.success(f"The Prediction Price of the Car is €{int(prediction)}")
            
        
if __name__ == '__main__':
    main()
