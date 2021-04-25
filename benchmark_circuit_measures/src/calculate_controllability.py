def calculateControllability(levels, circuitDescription, testability):
    for level in levels:
        for lineInd in level:
            if testability.get(lineInd) is None:
                testability[lineInd] = {}
            lineInfo = level[lineInd]
            control0 = calculateControllabilityToZero(lineInfo, circuitDescription, testability)
            testability[lineInd]["control0"] = control0
            control1 = calculateControllabilityToOne(lineInfo, circuitDescription, testability)
            testability[lineInd]["control1"] = control1


def calculateControllabilityToZero(lineInfo, circuitDescription, testability):
    gate = lineInfo["leaving"]
    if gate == None:
        control0 = 1
        return control0
    else:
        control0 = 0
        gateType = circuitDescription[2][gate]["type"].lower()
        if  gateType == "or":

            for line in circuitDescription[2][gate]["inputs"]:
                control0 = testability[line]["control0"] + control0
            control0 = control0 + 1
            return control0
        elif gateType== "nand":
            for line in circuitDescription[2][gate]["inputs"]:
                control0 = testability[line]["control1"] + control0
            control0 = control0 + 1
            return control0
        elif gateType == "fanout":
            line = circuitDescription[2][gate]["inputs"][0]
            control0 = testability[line]["control0"]
            return control0
        elif gateType == "and" :
            line = circuitDescription[2][gate]["inputs"][0]
            control0 = testability[line]["control0"]
            for line in circuitDescription[2][gate]["inputs"]:
                control0 = min(control0, testability[line]["control0"])
            control0 = control0 + 1
            return control0
        elif gateType == "nor":
            line = circuitDescription[2][gate]["inputs"][0]
            control0 = testability[line]["control1"]
            for line in circuitDescription[2][gate]["inputs"]:
                control0 = min(control0, testability[line]["control1"])
            control0 = control0 + 1
            return control0
        elif gateType == "not":
            line = circuitDescription[2][gate]["inputs"][0]
            control0 = testability[line]["control0"] + 1
            return control0
        elif gateType == "buf" or gateType == "buff":
            line = circuitDescription[2][gate]["inputs"][0]
            control0 = testability[line]["control0"]
            return control0

        #TODO review xor
        elif gateType == "xor":
            control0a=0
            control1a = 0
            for line in circuitDescription[2][gate]["inputs"]:
                control0a = testability[line]["control0"] + control0a

            for line in circuitDescription[2][gate]["inputs"]:
                control1a = testability[line]["control1"] + control1a
            control0=min(control0a,control1a)+1
            return control0
        else:
            print("error no gateType:" + gateType)

def calculateControllabilityToOne(lineInfo, circuitDescription, testability):
    gate = lineInfo["leaving"]
    if gate == None:
        control1 = 1
        return control1
    else:
        control1 = 0
        gateType = circuitDescription[2][gate]["type"].lower()

        if gateType == "and":
            for line in circuitDescription[2][gate]["inputs"]:
                control1 = testability[line]["control1"] + control1
            control1 = control1 + 1
            return control1
        elif gateType =="nor":
            for line in circuitDescription[2][gate]["inputs"]:
                control1 = testability[line]["control0"] + control1
            control1 = control1 + 1
            return control1
        elif gateType == "fanout":
            line = circuitDescription[2][gate]["inputs"][0]
            control1 = testability[line]["control1"]
            return control1
        elif  gateType == "or":
            line = circuitDescription[2][gate]["inputs"][0]
            control1 = testability[line]["control1"]
            for line in circuitDescription[2][gate]["inputs"]:
                control1 = min(control1, testability[line]["control1"])
            control1 = control1 + 1
            return control1
        elif gateType== "nand":
            line = circuitDescription[2][gate]["inputs"][0]
            control1 = testability[line]["control0"]
            for line in circuitDescription[2][gate]["inputs"]:
                control1 = min(control1, testability[line]["control0"])
            control1 = control1 + 1
            return control1
        elif gateType == "not":
            line = circuitDescription[2][gate]["inputs"][0]
            control1 = testability[line]["control1"] + 1
            return control1
        elif gateType == "buf" or gateType == "buff":
            line = circuitDescription[2][gate]["inputs"][0]
            control1 = testability[line]["control1"]
            return control1
        elif gateType =='xor':
            line = circuitDescription[2][gate]["inputs"][0]
            c01=testability[line]["control1"]
            c00=testability[line]["control0"]
            line2 = circuitDescription[2][gate]["inputs"][1]
            c10 = testability[line2]["control0"]
            c11 = testability[line2]["control1"]
            control1=min((c10+c01),(c00+c11))+1
            return control1

        else:
            print("error no gateType:" + gateType)
