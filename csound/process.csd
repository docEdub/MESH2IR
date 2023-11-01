<CsoundSynthesizer>
<CsInstruments>

// Notes:
// - Switching RIRs 5 times a second will make getting through all 900 RIRs take 3 minutes.

sr = 44000
kr = 48000
nchnls = 1
0dbfs = 1

gSIrPath = "../evaluate/Output/embeddings_0/S1_48k"

giIrTableIds[] init 901
giI init 0
while (giI < lenarray(giIrTableIds)) do
    SIrFilename = sprintf("%s/%d.wav", gSIrPath, giI + 1)
    giIrTableIds[giI] = ftgen(0, 0, 0, -1, SIrFilename, 0, 0, 0)
    giI += 1
od

giIrTableId = ftgen(0, 0, 0, -1, sprintf("%s/1.wav", gSIrPath), 0, 0, 0)

instr 1
    //aLeft, aRight diskin2 "aria_48k.wav", 1
    aLeft, aRight diskin2 "alarm-clock_48k.wav", 1, 0, 1

    kIrTableIdIndex init 0
    if (kIrTableIdIndex < lenarray(giIrTableIds) - 1) then
        kTime = times:k()
        kLastIrTime init 0
        kIrDuration = kTime - kLastIrTime

        kUpdateIr = 0

        if (kIrDuration > 0.025) then // switch 40 times a second
            kIrTableIdIndex += 1
            
            event("i", "SetIrTable", 0, 1, kIrTableIdIndex)
            kUpdateIr = 1

            kLastIrTime = kTime
        endif
    endif

    aOut = liveconv(aLeft, giIrTableId, 2048, kUpdateIr, 0)

    kthresh = -90
    kloknee = -15
    khiknee = -5
    kratio = 3
    katt = 0.1
    krel = 0.1
    ilook = 0.02
    aOut = compress2(aOut, aOut, kthresh, kloknee, khiknee, kratio, katt, krel, ilook)

    out aOut
endin

instr SetIrTable
    iIrTableIdIndex = p4
    tableicopy(giIrTableId, giIrTableIds[iIrTableIdIndex])

    iMaxValue = 0
    ii = 0
    while (ii < ftlen(giIrTableId)) do
        iMaxValue = max(iMaxValue, abs(tab_i(ii, giIrTableId)))
        ii += 1
    od
    prints("SetIrTable: [%d] max = %f\n", iIrTableIdIndex, iMaxValue)

    turnoff
endin
             
</CsInstruments>
<CsScore>

//i1 0 300
i1 0 30

</CsScore>
</CsoundSynthesizer>
