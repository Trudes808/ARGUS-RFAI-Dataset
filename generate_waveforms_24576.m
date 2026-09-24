function manifest = generate_waveforms_24576(varargin)
%GENERATE_WAVEFORMS_24576 Build a 245.76 MSps waveform library for SDR TX.
%
% Generates BPSK, QPSK, 16QAM, Broadband FM, Narrowband FM, OFDM, Bluetooth,
% 5G NR downlink, and IEEE 802.11ax waveforms (with parameter variations) at a
% common output sample rate of 245.76 MSps. Every waveform is produced at its
% natural/native sample rate and then RATIONALLY RESAMPLED to 245.76 MSps, so
% the stored P/Q factors invert the resampling exactly on the receive side.
%
% Output contract (every <name>.mat):
%   f_sig     complex single column vector @ 245.76e6  (complex64 IQ for Ettus)
%   Fs        output sample rate, always 245.76e6
%   txBits    PN9 payload bits (uint8) when the modulator accepts explicit bits
%             (empty for the analog audio-FM classes)
%   metadata  struct mirroring the adjacent <name>.json sidecar
%
% Bit source : ITU-T PN9 (x^9 + x^5 + 1, all-ones init) for every digital class.
% Audio source: Disco_Snail_easter_egg.mp3 drives both FM classes.
%
% Examples:
%   generate_waveforms_24576("Mode","smoke")   % small, fast self-test set
%   generate_waveforms_24576                    % full variation grid
%
% See decode_waveforms_24576.m for the matching receive / BER / audio chain.

opts = parseOptions(varargin{:});
rng(opts.RandomSeed, "twister");

if ~exist(opts.OutputRoot, "dir")
    mkdir(opts.OutputRoot);
end

records = {};
fprintf("Writing 245.76 MSps waveform library to %s\n", opts.OutputRoot);
fprintf("Output sample rate : %.2f MHz\n", opts.OutputSampleRateHz/1e6);
fprintf("Mode               : %s\n", opts.Mode);
fprintf("Audio source       : %s\n\n", opts.AudioFile);

records = generateSingleCarrierClass("BPSK",  2,  opts, records);
records = generateSingleCarrierClass("QPSK",  4,  opts, records);
records = generateSingleCarrierClass("16QAM", 16, opts, records);
records = generateGenericOFDM(opts, records);
records = generateBluetooth(opts, records);
records = generate5GDownlink(opts, records);
records = generateWLAN80211ax(opts, records);
records = generateAudioFM("Broadband_FM",  opts, records);
records = generateAudioFM("Narrowband_FM", opts, records);

if isempty(records)
    manifest = table();
else
    manifest = struct2table([records{:}]);
end

manifestPath = fullfile(opts.OutputRoot, "waveform_manifest.csv");
writetable(manifest, manifestPath);
writeReadme(opts, manifest);

fprintf("\nWrote %d waveforms\n", height(manifest));
fprintf("Manifest: %s\n", manifestPath);
end

% ======================================================================= %
%                               OPTIONS                                   %
% ======================================================================= %
function opts = parseOptions(varargin)
here = fileparts(mfilename("fullpath"));
p = inputParser;
p.addParameter("OutputRoot", fullfile(here, "generated_waveforms_24576"));
p.addParameter("OutputSampleRateHz", 245.76e6);
p.addParameter("AudioFile", fullfile(here, "Disco_Snail_easter_egg.mp3"));
p.addParameter("Mode", "full");
p.addParameter("RandomSeed", 42);
p.addParameter("SingleCarrierSymbols", 2048);
p.addParameter("FMDurationSeconds", 0.020);   % <= 25 ms cap
p.addParameter("AudioStartSeconds", 2.0);      % skip MP3 leading silence
p.parse(varargin{:});
opts = p.Results;

opts.Mode = string(opts.Mode);
opts.Smoke = strcmpi(opts.Mode, "smoke");
opts.AudioFile = char(opts.AudioFile);

% ---- single-carrier variation grid ----
% symbol rates 0.24 -> 61.44 MHz in factors of four
opts.SymbolRatesHz = [0.24e6 0.96e6 3.84e6 15.36e6 61.44e6];
% native filter oversampling factor (samples/symbol BEFORE rational resample)
opts.FilterOversampling = [4 8];
opts.SingleCarrierFilters = [ ...
    struct("Name","rrc_a0p20", "Shape","Square root", "Rolloff",0.20, "Span",10), ...
    struct("Name","rrc_a0p35", "Shape","Square root", "Rolloff",0.35, "Span",10), ...
    struct("Name","rc_a0p35",  "Shape","Normal",      "Rolloff",0.35, "Span",10) ];

% ---- OFDM (OFDMEndToEndExample helpers: BWIndex / modOrder / codeRateIndex) ----
opts.OFDM_BWIndex        = [3 4 5 7];   % 5, 10, 20, 98.28 MHz (FFT 512/1024/2048/4096)
opts.OFDM_ModOrders      = [4 16 64];   % QPSK, 16QAM, 64QAM
opts.OFDM_CodeRateIndex  = [0 2];       % 1/2, 3/4
opts.OFDM_NumSymPerFrame = 30;
opts.OFDM_NumFrames      = 5;   % >= ceil(144/numSymPerFrame); a few decode after camping

% ---- Bluetooth ----
opts.BTSamplesPerSymbol = [8 16];

% ---- 5G NR downlink ----
opts.NR_BW_MHz = [5 10 20 25 50 100];
opts.NR_SCS    = [15 30];
opts.NR_Mods   = ["QPSK" "16QAM" "64QAM" "256QAM"];

% ---- 802.11ax ----
opts.WLAN_CBW = ["CBW20" "CBW40" "CBW80" "CBW160"];
opts.WLAN_MCS = [0 4 7 9];
opts.WLAN_APEPLength = 512;

% ---- FM ----
opts.BroadbandDevHz  = [50e3 75e3 1e6 10e6 30.72e6];
opts.BroadbandAudioLPFHz = 15e3;
opts.NarrowbandDevHz = [1e3 2.5e3 5e3 10e3];
opts.NarrowbandAudioLPFHz = 3e3;

if opts.Smoke
    opts.SymbolRatesHz = [0.96e6 61.44e6];
    opts.FilterOversampling = 4;
    opts.SingleCarrierFilters = opts.SingleCarrierFilters([1 3]);
    opts.SingleCarrierSymbols = 512;
    opts.OFDM_BWIndex = 4;
    opts.OFDM_ModOrders = [4 16];
    opts.OFDM_CodeRateIndex = 0;
    opts.BTSamplesPerSymbol = 8;
    opts.NR_BW_MHz = [10 50];
    opts.NR_SCS = 30;
    opts.NR_Mods = ["QPSK" "16QAM"];
    opts.WLAN_CBW = ["CBW20" "CBW80"];
    opts.WLAN_MCS = [0 7];
    opts.BroadbandDevHz = [75e3 30.72e6];
    opts.NarrowbandDevHz = [2.5e3 10e3];
    opts.FMDurationSeconds = 0.010;
end
end

% ======================================================================= %
%                       SINGLE CARRIER (BPSK/QPSK/16QAM)                   %
% ======================================================================= %
function records = generateSingleCarrierClass(className, M, opts, records)
folder = ensureClassFolder(opts.OutputRoot, className);
settings = singleCarrierSettings(className);
if opts.Smoke
    settings = settings(1);
end
bitsPerSymbol = log2(M);

for Rs = opts.SymbolRatesHz
    for fosf = opts.FilterOversampling
        nativeFs = Rs * fosf;                 % native samp rate before resample
        for setting = settings
            for pulse = opts.SingleCarrierFilters
                bits = pn9Bits(opts.SingleCarrierSymbols * bitsPerSymbol);
                syms = modulateSingleCarrier(bits, M, setting);

                txFilter = comm.RaisedCosineTransmitFilter( ...
                    "Shape", pulse.Shape, ...
                    "RolloffFactor", pulse.Rolloff, ...
                    "FilterSpanInSymbols", pulse.Span, ...
                    "OutputSamplesPerSymbol", fosf);
                nativeWave = txFilter(syms);
                release(txFilter);

                [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
                wave = capDuration(wave, opts.OutputSampleRateHz, 0.025);
                wave = normalizeWaveform(wave);

                occBw = Rs * (1 + pulse.Rolloff);
                name = joinTags(className, "Rs" + formatHzTag(Rs), ...
                    "fosf" + fosf, "occ" + formatHzTag(occBw), ...
                    setting.Name, pulse.Name, "pn9");

                metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
                metadata.standard = "Generic single-carrier";
                metadata.bitSource = "PN9";
                metadata.payloadBitsStored = true;
                metadata.designedOccupiedBandwidthHz = occBw;
                metadata.requestedOccupiedBandwidthHz = occBw;
                metadata.nativeSampleRateHz = nativeFs;
                metadata.symbolRateHz = Rs;
                metadata.modulation = char(className);
                metadata.modulationSetting = setting.Description;
                metadata.pulseShape = pulse.Shape;
                metadata.rolloff = pulse.Rolloff;
                metadata.filterSpanSymbols = pulse.Span;
                metadata.samplesPerSymbol = fosf;        % native samples/symbol
                metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
                metadata.variationsExplored = sprintf( ...
                    "symbolRate=%gMHz; filterOversampling=%d; rolloff=%g; filterShape=%s", ...
                    Rs/1e6, fosf, pulse.Rolloff, pulse.Shape);
                metadata.singleCarrier = struct( ...
                    "ModulationOrder", M, "BitsPerSymbol", bitsPerSymbol, ...
                    "PhaseOffsetRad", setting.PhaseOffset, "SymbolOrder", setting.SymbolOrder, ...
                    "FilterShape", pulse.Shape, "RolloffFactor", pulse.Rolloff, ...
                    "FilterSpanInSymbols", pulse.Span, "SamplesPerSymbol", fosf);
                metadata.decodingRecipe = "Resample 245.76 MSps -> nativeSampleRateHz with Q/P, matched-filter with the stored raised-cosine settings (DecimationFactor=samplesPerSymbol), align to PN9, equalize complex gain, demodulate, compare to txBits.";

                records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
            end
        end
    end
end
end

function settings = singleCarrierSettings(className)
switch string(className)
    case "BPSK"
        settings = [ ...
            struct("Name","phase0_gray",   "PhaseOffset",0,    "SymbolOrder","gray", "Description","BPSK phaseOffset 0 rad, gray order"), ...
            struct("Name","phasepi4_gray", "PhaseOffset",pi/4, "SymbolOrder","gray", "Description","BPSK phaseOffset pi/4 rad, gray order") ];
    case "QPSK"
        settings = [ ...
            struct("Name","phasepi4_gray", "PhaseOffset",pi/4, "SymbolOrder","gray", "Description","QPSK phaseOffset pi/4 rad, gray order"), ...
            struct("Name","phase0_gray",   "PhaseOffset",0,    "SymbolOrder","gray", "Description","QPSK phaseOffset 0 rad, gray order") ];
    case "16QAM"
        settings = [ ...
            struct("Name","gray", "PhaseOffset",0, "SymbolOrder","gray", "Description","16QAM gray symbol order, unit average power"), ...
            struct("Name","bin",  "PhaseOffset",0, "SymbolOrder","bin",  "Description","16QAM binary symbol order, unit average power") ];
    otherwise
        error("Unsupported single-carrier class %s", className);
end
end

function syms = modulateSingleCarrier(bits, M, setting)
bits = double(bits(:));
switch M
    case {2,4}
        syms = pskmod(bits, M, setting.PhaseOffset, setting.SymbolOrder, "InputType","bit");
    otherwise
        syms = qammod(bits, M, setting.SymbolOrder, "InputType","bit", "UnitAveragePower",true);
end
syms = syms(:);
end

% ======================================================================= %
%                           GENERIC OFDM                                  %
% ======================================================================= %
% Uses the MathWorks "OFDM Transmitter and Receiver" example helpers
% (OFDMEndToEndExample): a full framed OFDM link (sync + reference + header +
% pilots + data, with CRC, convolutional coding, puncturing, and interleaving).
% Varies the example's system parameters: BWIndex, modulation order, code rate.
function records = generateGenericOFDM(opts, records)
className = "OFDM";
folder = ensureClassFolder(opts.OutputRoot, className);
ensureOFDMHelpers();

for bwi = opts.OFDM_BWIndex
    for modOrder = opts.OFDM_ModOrders
        for cri = opts.OFDM_CodeRateIndex
            userParam = ofdmUserParam(bwi, modOrder, cri, ...
                opts.OFDM_NumSymPerFrame, opts.OFDM_NumFrames);
            [sysParam, txParam] = helperOFDMSetParameters(userParam);
            txObj = helperOFDMTxInit(sysParam);

            % Same PN9 transport block in every frame, so any decoded frame can
            % be compared against the known bits without frame-index tracking.
            block = double(pn9Bits(sysParam.trBlkSize));
            txParam.txDataBits = block;

            nativeWave = [];
            for fr = 1:sysParam.numFrames
                sysParam.frameNum = fr;
                txOut = helperOFDMTx(txParam, sysParam, txObj);
                nativeWave = [nativeWave; txOut(:)]; %#ok<AGROW>
            end

            nativeFs = sysParam.scs * sysParam.FFTLen;
            [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
            wave = normalizeWaveform(wave);

            occBw = sysParam.usedSubCarr * sysParam.scs;
            frameLen = (sysParam.FFTLen + sysParam.CPLen) * sysParam.numSymPerFrame;
            crName = ofdmCodeRateName(cri);
            name = joinTags(className, "bwidx" + bwi, "occ" + formatHzTag(occBw), ...
                "fft" + sysParam.FFTLen, "mod" + modOrder, "cr" + crName, "pn9");

            metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
            metadata.standard = "Generic OFDM";
            metadata.bitSource = "PN9";
            metadata.payloadBitsStored = true;
            metadata.designedOccupiedBandwidthHz = occBw;
            metadata.requestedOccupiedBandwidthHz = sysParam.BW;
            metadata.nativeSampleRateHz = nativeFs;
            metadata.symbolRateHz = sysParam.scs;
            metadata.modulation = "OFDM-" + modOrderName(modOrder);
            metadata.modulationSetting = sprintf("helperOFDM BWIndex %d, %s subcarriers, codeRate %s", ...
                bwi, modOrderName(modOrder), strrep(crName,"_","/"));
            metadata.pulseShape = "helperOFDM framed OFDM (sync+ref+header+pilots+data, CP)";
            metadata.rolloff = NaN;
            metadata.samplesPerSymbol = NaN;
            metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
            metadata.variationsExplored = sprintf("BWIndex=%d; modOrder=%d; codeRateIndex=%d (%s)", ...
                bwi, modOrder, cri, strrep(crName,"_","/"));
            metadata.ofdm = struct( ...
                "Source", "OFDMEndToEndExample helperOFDM*", ...
                "BWIndex", bwi, "ModOrder", modOrder, "CodeRateIndex", cri, ...
                "NumFrames", sysParam.numFrames, "NumSymPerFrame", sysParam.numSymPerFrame, ...
                "Fc", userParam.fc, "TrBlkSize", sysParam.trBlkSize, ...
                "FFTLen", sysParam.FFTLen, "CPLen", sysParam.CPLen, ...
                "SCS", sysParam.scs, "UsedSubCarr", sysParam.usedSubCarr, ...
                "BW", sysParam.BW, "NativeSampleRateHz", nativeFs, ...
                "FrameLenSamples", frameLen);
            metadata.decodingRecipe = "addpath the OFDMEndToEndExample helpers, resample to NativeSampleRateHz, drive helperOFDMRxFrontEnd + helperOFDMRx frame-by-frame with timingAdvance feedback, and compare each CRC-passing frame's trBlkSize bits to txBits.";

            records = saveWaveformRecord(records, folder, name, wave, block, metadata, opts);
        end
    end
end
end

function userParam = ofdmUserParam(bwIndex, modOrder, codeRateIndex, numSymPerFrame, numFrames)
userParam = struct( ...
    "BWIndex", bwIndex, "modOrder", modOrder, "codeRateIndex", codeRateIndex, ...
    "numFrames", numFrames, "numSymPerFrame", numSymPerFrame, "fc", 0, ...
    "enableCFO", false, "enableCPE", true, "enableFading", false, ...
    "chanVisual", false, "enableScopes", false, "verbosity", 0);
end

function name = ofdmCodeRateName(cri)
switch cri
    case 0, name = "1_2";
    case 1, name = "2_3";
    case 2, name = "3_4";
    case 3, name = "5_6";
    otherwise, name = "1_2";
end
end

function ensureOFDMHelpers()
% Portable locator for the MathWorks "OFDM Transmitter and Receiver" example
% helpers. Order: already-on-path -> OFDM_HELPER_DIR env var -> ~/Documents/
% MATLAB/Examples/*/comm/OFDMEndToEndExample -> matlabroot examples.
if exist("helperOFDMTx","file") == 2, return; end
cands = string.empty;
e = string(getenv("OFDM_HELPER_DIR"));
if strlength(e) > 0, cands(end+1) = e; end
home = getenv("HOME"); if isempty(home), home = getenv("USERPROFILE"); end
d = dir(fullfile(home, "Documents", "MATLAB", "Examples", "*", "comm", "OFDMEndToEndExample"));
for i = 1:numel(d), cands(end+1) = string(fullfile(d(i).folder, d(i).name)); end
cands(end+1) = string(fullfile(matlabroot, "examples", "comm", "OFDMEndToEndExample"));
for c = cands
    if isfolder(c), addpath(char(c)); end
    if exist("helperOFDMTx","file") == 2, return; end
end
error("generate_waveforms_24576:OFDMHelpers", ...
    ["helperOFDM* not found. Run openExample('comm/OFDMEndToEndExample') once " ...
     "to install the example, or set the OFDM_HELPER_DIR environment variable to " ...
     "its folder before calling this function."]);
end

% ======================================================================= %
%                             BLUETOOTH                                   %
% ======================================================================= %
function records = generateBluetooth(opts, records)
className = "Bluetooth";
folder = ensureClassFolder(opts.OutputRoot, className);

brSpecs = [ ...
    struct("Mode","BR",    "PacketType","DH1",   "PayloadBytes",27, "SymbolRateHz",1e6, "OccupiedHz",1e6), ...
    struct("Mode","EDR2M", "PacketType","2-DH1", "PayloadBytes",54, "SymbolRateHz",1e6, "OccupiedHz",1e6), ...
    struct("Mode","EDR3M", "PacketType","3-DH1", "PayloadBytes",83, "SymbolRateHz",1e6, "OccupiedHz",1e6) ];
bleSpecs = [ ...
    struct("Mode","LE1M",   "SymbolRateHz",1e6, "OccupiedHz",1e6), ...
    struct("Mode","LE2M",   "SymbolRateHz",2e6, "OccupiedHz",2e6), ...
    struct("Mode","LE500K", "SymbolRateHz",1e6, "OccupiedHz",1e6), ...
    struct("Mode","LE125K", "SymbolRateHz",1e6, "OccupiedHz",1e6) ];
if opts.Smoke
    brSpecs = brSpecs(1);
    bleSpecs = bleSpecs([1 2]);
end

for sps = opts.BTSamplesPerSymbol
    for spec = brSpecs
        nativeFs = sps * spec.SymbolRateHz;
        cfg = bluetoothWaveformConfig("Mode",spec.Mode, "PacketType",spec.PacketType, ...
            "PayloadLength",spec.PayloadBytes, "SamplesPerSymbol",sps, "WhitenStatus","On");
        bits = pn9Bits(spec.PayloadBytes*8);
        nativeWave = bluetoothWaveformGenerator(double(bits), cfg);
        [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
        wave = normalizeWaveform(wave);

        name = joinTags(className, lower(spec.Mode), lower(strrep(spec.PacketType,"-","")), "sps"+sps, "pn9");
        metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
        metadata.standard = "Bluetooth BR/EDR";
        metadata.bitSource = "PN9";
        metadata.payloadBitsStored = true;
        metadata.designedOccupiedBandwidthHz = spec.OccupiedHz;
        metadata.requestedOccupiedBandwidthHz = spec.OccupiedHz;
        metadata.nativeSampleRateHz = nativeFs;
        metadata.symbolRateHz = spec.SymbolRateHz;
        metadata.modulation = spec.Mode;
        metadata.modulationSetting = spec.PacketType + ", whitening on";
        metadata.pulseShape = "Bluetooth BR/EDR Gaussian / EDR pulse shaping";
        metadata.rolloff = NaN;
        metadata.samplesPerSymbol = sps;
        metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
        metadata.variationsExplored = sprintf("mode=%s; packetType=%s; samplesPerSymbol=%d", ...
            spec.Mode, spec.PacketType, sps);
        metadata.standardConfig = struct("Mode",spec.Mode, "PacketType",spec.PacketType, ...
            "PayloadLengthBytes",spec.PayloadBytes, "SamplesPerSymbol",sps, "WhitenStatus","On");
        metadata.decodingRecipe = "Resample to native, bluetoothIdealReceiver with stored Mode/SamplesPerSymbol/whitening, compare payload bits to txBits.";
        records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
    end

    for spec = bleSpecs
        nativeFs = sps * spec.SymbolRateHz;
        bits = pn9Bits(8*32);     % 32-byte payload
        nativeWave = bleWaveformGenerator(double(bits), "Mode",spec.Mode, ...
            "SamplesPerSymbol",sps, "WhitenStatus","On", "ModulationIndex",0.5, "PulseLength",1);
        [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
        wave = normalizeWaveform(wave);

        name = joinTags(className, lower(spec.Mode), "sps"+sps, "pn9");
        metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
        metadata.standard = "Bluetooth LE";
        metadata.bitSource = "PN9";
        metadata.payloadBitsStored = true;
        metadata.designedOccupiedBandwidthHz = spec.OccupiedHz;
        metadata.requestedOccupiedBandwidthHz = spec.OccupiedHz;
        metadata.nativeSampleRateHz = nativeFs;
        metadata.symbolRateHz = spec.SymbolRateHz;
        metadata.modulation = spec.Mode;
        metadata.modulationSetting = spec.Mode + ", whitening on, modIndex 0.5, pulseLength 1";
        metadata.pulseShape = "Bluetooth LE Gaussian frequency pulse";
        metadata.rolloff = NaN;
        metadata.samplesPerSymbol = sps;
        metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
        metadata.variationsExplored = sprintf("mode=%s; samplesPerSymbol=%d", spec.Mode, sps);
        metadata.standardConfig = struct("Mode",spec.Mode, "PayloadBytes",32, ...
            "SamplesPerSymbol",sps, "WhitenStatus","On", "ModulationIndex",0.5, "PulseLength",1);
        metadata.decodingRecipe = "Resample to native, bleIdealReceiver with stored Mode/SamplesPerSymbol/whitening/modIndex/pulseLength, compare payload bits to txBits.";
        records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
    end
end
end

% ======================================================================= %
%                          5G NR DOWNLINK                                 %
% ======================================================================= %
function records = generate5GDownlink(opts, records)
className = "5G_Downlink";
folder = ensureClassFolder(opts.OutputRoot, className);

for bw = opts.NR_BW_MHz
    for scs = opts.NR_SCS
        nrb = nrFr1Nrb(bw, scs);
        if isnan(nrb), continue; end
        refCarrier = nrCarrierConfig("NCellID",1, "NSizeGrid",nrb, "SubcarrierSpacing",scs, "NSlot",0);
        nativeFs = nrOFDMInfo(refCarrier).SampleRate;

        for modName = opts.NR_Mods
            cfg = nrDLCarrierConfig;
            cfg.FrequencyRange = "FR1";
            cfg.ChannelBandwidth = bw;
            cfg.NumSubframes = 1;
            cfg.SampleRate = nativeFs;       % generate at NATIVE NR rate
            cfg.CarrierFrequency = 0;
            cfg.NCellID = 1;
            cfg.WindowingPercent = 0;

            carrier = nrSCSCarrierConfig("SubcarrierSpacing",scs, "NSizeGrid",nrb, "NStartGrid",0);
            cfg.SCSCarriers = {carrier};
            bwp = nrWavegenBWPConfig("BandwidthPartID",1, "SubcarrierSpacing",scs, "NSizeBWP",nrb, "NStartBWP",0);
            bwp.Label = "BWP1";
            cfg.BandwidthParts = {bwp};

            slotsPerSubframe = scs/15;
            numSlots = slotsPerSubframe * cfg.NumSubframes;   % fill every slot
            codeRate = targetCodeRateFor5G(modName);
            pdsch = nrWavegenPDSCHConfig;
            pdsch.Enable = true;
            pdsch.Label = "PDSCH_PN9";
            pdsch.BandwidthPartID = 1;
            pdsch.Modulation = char(modName);
            pdsch.NumLayers = 1;
            pdsch.PRBSet = 0:(nrb-1);
            pdsch.SymbolAllocation = [0 14];
            pdsch.SlotAllocation = 0:(slotsPerSubframe-1);     % PDSCH in all slots
            pdsch.Period = slotsPerSubframe;
            pdsch.TargetCodeRate = codeRate;
            pdsch.RVSequence = 0;

            refPDSCH = nrPDSCHConfig("Modulation",char(modName), "NumLayers",1, ...
                "PRBSet",0:(nrb-1), "SymbolAllocation",[0 14]);
            refPDSCH.RNTI = 1;
            [~, refInfo] = nrPDSCHIndices(refCarrier, refPDSCH);
            tbs = nrTBS(char(modName), 1, nrb, refInfo.NREPerPRB, codeRate, 0);
            bits = pn9Bits(tbs * numSlots);                    % one TB per slot
            pdsch.DataSource = bits;
            cfg.PDSCH = {pdsch};
            try, cfg.SSBurst.Enable = false; catch, end
            try, cfg.PDCCH{1}.Enable = false; catch, end
            try, cfg.CSIRS{1}.Enable = false; catch, end

            [nativeWave, info] = nrWaveformGenerator(cfg);
            [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
            wave = normalizeWaveform(wave);
            occBw = 12*nrb*scs*1e3;

            name = joinTags(className, "bw"+formatHzTag(bw*1e6), "scs"+scs+"kHz", "pdsch"+modName, "pn9");
            metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
            metadata.standard = "5G NR downlink";
            metadata.bitSource = "PN9";
            metadata.payloadBitsStored = true;
            metadata.designedOccupiedBandwidthHz = occBw;
            metadata.requestedOccupiedBandwidthHz = bw*1e6;
            metadata.nativeSampleRateHz = nativeFs;
            metadata.symbolRateHz = NaN;
            metadata.modulation = char(modName);
            metadata.modulationSetting = "PDSCH " + modName + ", targetCodeRate " + codeRate;
            metadata.pulseShape = "CP-OFDM";
            metadata.rolloff = NaN;
            metadata.samplesPerSymbol = NaN;
            metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
            metadata.variationsExplored = sprintf("channelBW=%dMHz; SCS=%dkHz; PDSCHmod=%s", bw, scs, modName);
            metadata.standardConfig = struct( ...
                "ChannelBandwidthMHz",bw, "SubcarrierSpacingkHz",scs, "NSizeGrid",nrb, ...
                "NumSubframes",cfg.NumSubframes, "NumSlots",numSlots, "PDSCHModulation",char(modName), ...
                "PDSCHPRBSet",[0 nrb-1], "PDSCHSymbolAllocation",pdsch.SymbolAllocation, ...
                "PDSCHSlotAllocation",[0 slotsPerSubframe-1], "RV",0, "TargetCodeRate",codeRate, ...
                "TransportBlockSize",tbs, "NativeSampleRateHz",nativeFs, ...
                "GeneratorInfoSampleRateHz", readStructField(info,"SampleRate",nativeFs));
            metadata.decodingRecipe = "Resample to NativeSampleRateHz, nrOFDMDemodulate, channel-estimate from PDSCH DM-RS, MMSE-equalize, nrPDSCHDecode + nrDLSCHDecoder with stored TBS/codeRate, compare to txBits.";
            records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
        end
    end
end
end

function rate = targetCodeRateFor5G(modName)
switch string(modName)
    case "QPSK",   rate = 0.4785;
    case "16QAM",  rate = 0.5137;
    case "64QAM",  rate = 0.6504;
    case "256QAM", rate = 0.7539;
    otherwise,     rate = 0.5137;
end
end

function nrb = nrFr1Nrb(channelBandwidthMHz, scsKHz)
% TS 38.101-1 FR1 max transmission bandwidth configuration (subset).
switch scsKHz
    case 15
        keys = [5 10 15 20 25 30 40 50];
        vals = [25 52 79 106 133 160 216 270];
    case 30
        keys = [5 10 15 20 25 30 40 50 60 80 90 100];
        vals = [11 24 38 51 65 78 106 133 162 217 245 273];
    otherwise
        nrb = NaN; return;
end
idx = find(keys == channelBandwidthMHz, 1);
if isempty(idx), nrb = NaN; else, nrb = vals(idx); end
end

% ======================================================================= %
%                            802.11ax                                     %
% ======================================================================= %
function records = generateWLAN80211ax(opts, records)
className = "802_11ax";
folder = ensureClassFolder(opts.OutputRoot, className);

for cbw = opts.WLAN_CBW
    for mcs = opts.WLAN_MCS
        cfg = wlanHESUConfig;
        cfg.ChannelBandwidth = char(cbw);
        cfg.MCS = mcs;
        cfg.ChannelCoding = "LDPC";
        cfg.APEPLength = opts.WLAN_APEPLength;
        cfg.GuardInterval = 3.2;
        cfg.HELTFType = 4;
        cfg.NumTransmitAntennas = 1;
        cfg.NumSpaceTimeStreams = 1;

        nativeFs = wlanSampleRate(cfg);
        psduLen = getPSDULength(cfg);     % PSDU length in bits
        bits = pn9Bits(psduLen);
        nativeWave = wlanWaveformGenerator(int8(bits), cfg, "NumPackets",1, ...
            "IdleTime", 0, "ScramblerInitialization",93);   % no trailing idle
        [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
        wave = normalizeWaveform(wave);

        occBw = wlanBandwidthHz(cbw);
        name = joinTags(className, "bw"+formatHzTag(occBw), "mcs"+mcs, "gi3p2us", "heltf4", "pn9");
        metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
        metadata.standard = "IEEE 802.11ax HE-SU";
        metadata.bitSource = "PN9";
        metadata.payloadBitsStored = true;
        metadata.designedOccupiedBandwidthHz = occBw;
        metadata.requestedOccupiedBandwidthHz = occBw;
        metadata.nativeSampleRateHz = nativeFs;
        metadata.symbolRateHz = NaN;
        metadata.modulation = "HE-MCS" + mcs;
        metadata.modulationSetting = "MCS " + mcs + ", LDPC, GI 3.2us, HE-LTF 4";
        metadata.pulseShape = "802.11ax OFDM";
        metadata.rolloff = NaN;
        metadata.samplesPerSymbol = NaN;
        metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
        metadata.variationsExplored = sprintf("channelBW=%s; MCS=%d", cbw, mcs);
        metadata.standardConfig = struct("ChannelBandwidth",char(cbw), "MCS",mcs, ...
            "ChannelCoding",char(cfg.ChannelCoding), "APEPLengthBytes",cfg.APEPLength, ...
            "GuardIntervalUs",cfg.GuardInterval, "HELTFType",cfg.HELTFType, ...
            "ScramblerInitialization",93, "PSDULengthBits",psduLen);
        metadata.decodingRecipe = "Resample to native, sync HE preamble, channel-estimate from HE-LTF, equalize, wlanHEDataBitRecover, compare to txBits.";
        records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
    end
end
end

function nbits = getPSDULength(cfg)
try
    psdu = psduLength(cfg);   % PSDU length in bytes (per user)
    nbits = double(psdu(1)) * 8;
catch
    nbits = double(cfg.APEPLength) * 8;
end
end

function bwHz = wlanBandwidthHz(cbw)
switch string(cbw)
    case "CBW20",  bwHz = 20e6;
    case "CBW40",  bwHz = 40e6;
    case "CBW80",  bwHz = 80e6;
    case "CBW160", bwHz = 160e6;
    otherwise, error("Unsupported WLAN bandwidth %s", cbw);
end
end

% ======================================================================= %
%                       AUDIO FM (mp3 source)                             %
% ======================================================================= %
function records = generateAudioFM(className, opts, records)
folder = ensureClassFolder(opts.OutputRoot, className);
[sourceAudio, sourceFs] = readAudioMono(opts.AudioFile);

switch string(className)
    case "Broadband_FM"
        devList = opts.BroadbandDevHz;  audioLPF = opts.BroadbandAudioLPFHz;
        standard = "MP3 audio-source broadband FM";
        desc = "FM-radio-style mono audio FM";
    case "Narrowband_FM"
        devList = opts.NarrowbandDevHz; audioLPF = opts.NarrowbandAudioLPFHz;
        standard = "MP3 audio-source narrowband FM";
        desc = "walkie-talkie-style voice FM";
    otherwise
        error("Unsupported FM class %s", className);
end

for freqDevHz = devList
    nativeFs = chooseFMNativeRate(opts.OutputSampleRateHz, freqDevHz, audioLPF);
    numNative = max(64, round(opts.FMDurationSeconds * nativeFs));
    audioNative = prepareAudioSegment(sourceAudio, sourceFs, min(audioLPF, 0.45*nativeFs), ...
        nativeFs, numNative, opts.AudioStartSeconds);

    phase = 2*pi*freqDevHz/nativeFs * cumsum(audioNative);
    nativeWave = exp(1j*phase);
    [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
    wave = normalizeWaveform(wave);

    carsonBw = 2*(freqDevHz + audioLPF);
    name = joinTags(className, "dev"+formatHzTag(freqDevHz), "occ"+formatHzTag(carsonBw), ...
        "lpf"+formatHzTag(audioLPF), "mp3");
    metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
    metadata.standard = standard;
    metadata.bitSource = "Not applicable; analog MP3 audio source";
    metadata.payloadBitsStored = false;
    metadata.designedOccupiedBandwidthHz = carsonBw;
    metadata.requestedOccupiedBandwidthHz = carsonBw;
    metadata.nativeSampleRateHz = nativeFs;
    metadata.symbolRateHz = NaN;
    metadata.modulation = "FM";
    metadata.modulationSetting = desc + ", peak deviation " + freqDevHz + " Hz, audio LPF " + audioLPF + " Hz";
    metadata.pulseShape = "Audio lowpass message shaping; no digital pulse shaping";
    metadata.rolloff = NaN;
    metadata.samplesPerSymbol = NaN;
    metadata.frequencyDeviationHz = freqDevHz;
    metadata.audioSourceFile = char(opts.AudioFile);
    metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
    metadata.variationsExplored = sprintf("frequencyDeviation=%gHz; audioLPF=%gHz", freqDevHz, audioLPF);
    metadata.audioFm = struct("AudioSourceFile",char(opts.AudioFile), ...
        "SourceAudioSampleRateHz",sourceFs, "AudioLowpassHz",min(audioLPF,0.45*nativeFs), ...
        "FrequencyDeviationHz",freqDevHz, "CarsonBandwidthHz",carsonBw, ...
        "DurationSeconds",numNative/nativeFs, "SourceStartSeconds",opts.AudioStartSeconds);
    metadata.decodingRecipe = "Resample to native, FM-demodulate by phase differentiation, audio-lowpass, resample to audio rate. Compare recovered audio to the band-limited source segment (audio SNR / correlation). BER not applicable.";
    metadata.note = "Analog FM audio: no bit payload, so BER is intentionally not computed.";

    records = saveWaveformRecord(records, folder, name, wave, uint8([]), metadata, opts);
end
end

function nativeFs = chooseFMNativeRate(outputFs, freqDevHz, audioLPF)
carson = 2*(freqDevHz + audioLPF);
minNative = max(480e3, 3*carson);
cands = integerDivisors(round(outputFs));
cands = cands(cands >= minNative & cands <= outputFs);
if isempty(cands)
    nativeFs = outputFs;
else
    nativeFs = cands(1);
end
end

function [audio, fs] = readAudioMono(audioFile)
[audio, fs] = audioread(audioFile);
if size(audio,2) > 1, audio = mean(audio,2); end
audio = double(audio(:));
audio(~isfinite(audio)) = 0;
pk = max(abs(audio));
if pk > 0, audio = audio./pk; end
end

function audioNative = prepareAudioSegment(audio, audioFs, audioLPF, nativeFs, numSamples, startSec)
audio = double(audio(:));
if audioLPF < 0.45*audioFs
    audio = lowpassAudio(audio, audioLPF, audioFs);
end
srcDur = numel(audio)/audioFs;
tTarget = (0:numSamples-1).'/nativeFs;
tSource = (0:numel(audio)-1).'/audioFs;
audioNative = interp1(tSource, audio, mod(startSec + tTarget, srcDur), "linear", 0);
audioNative = audioNative(:) - mean(audioNative);
pk = max(abs(audioNative));
if pk > 0, audioNative = 0.95*audioNative./pk; end
end

function y = lowpassAudio(x, cutoffHz, fs)
cutoffHz = min(cutoffHz, 0.45*fs);
try
    y = lowpass(x, cutoffHz, fs, "Steepness",0.85);
catch
    [b,a] = butter(6, cutoffHz/(fs/2));
    y = filtfilt(b,a,x);
end
end

% ======================================================================= %
%                          SHARED HELPERS                                 %
% ======================================================================= %
function syms = modulateBits(bits, M)
bits = double(bits(:));
switch M
    case 2
        syms = pskmod(bits, 2, 0, "gray", "InputType","bit");
    case 4
        syms = pskmod(bits, 4, pi/4, "gray", "InputType","bit");
    otherwise
        syms = qammod(bits, M, "gray", "InputType","bit", "UnitAveragePower",true);
end
syms = syms(:);
end

function name = modOrderName(M)
switch M
    case 2,  name = "BPSK";
    case 4,  name = "QPSK";
    case 16, name = "16QAM";
    case 64, name = "64QAM";
    case 256,name = "256QAM";
    otherwise, name = string(M)+"QAM";
end
end

function [wave, p, q] = resampleToTarget(nativeWave, nativeFs, targetFs)
[p, q] = rat(targetFs/nativeFs, 1e-12);
if p == q
    wave = double(nativeWave(:));
else
    wave = resample(double(nativeWave(:)), p, q, 20);   % order-20 Kaiser for flat passband
end
end

function y = capDuration(x, Fs, maxSeconds)
maxN = floor(maxSeconds*Fs);
if numel(x) > maxN
    y = x(1:maxN);
else
    y = x;
end
end

function y = normalizeWaveform(x)
x = x(:);
scale = sqrt(mean(abs(x).^2));
if isfinite(scale) && scale > 0
    x = 0.8*x/scale;
end
y = complex(single(real(x)), single(imag(x)));
end

function bits = pn9Bits(n)
% ITU-T O.150 PN9: x^9 + x^5 + 1, all-ones initial state.
state = true(1,9);
bits = false(n,1);
for idx = 1:n
    bits(idx) = state(9);
    newBit = xor(state(5), state(9));
    state = [newBit state(1:8)];
end
bits = uint8(bits);
end

function values = integerDivisors(n)
values = [];
limit = floor(sqrt(double(n)));
for c = 1:limit
    if mod(n,c) == 0
        values(end+1) = c;       %#ok<AGROW>
        values(end+1) = n/c;     %#ok<AGROW>
    end
end
values = sort(unique(values));
end

function folder = ensureClassFolder(outputRoot, className)
folder = fullfile(outputRoot, char(className));
if ~exist(folder, "dir"), mkdir(folder); end
end

function metadata = baseMetadata(className, waveformName, outputFs)
metadata = struct;
metadata.waveformName = char(waveformName);
metadata.class = char(className);
metadata.outputSampleRateHz = outputFs;
metadata.centerFrequencyHz = 0;
metadata.outputVariable = "f_sig";
metadata.outputDataType = "complex single (complex64 IQ)";
metadata.createdBy = "generate_waveforms_24576.m";
metadata.createdOn = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss ZZZZ"));
metadata.pn9Polynomial = "x^9 + x^5 + 1";
metadata.pn9InitialState = "all ones";
end

function records = saveWaveformRecord(records, folder, waveformName, wave, txBits, metadata, opts, codedBits)
safeName = sanitizeName(waveformName);
matPath = fullfile(folder, safeName + ".mat");
jsonPath = fullfile(folder, safeName + ".json");

f_sig = wave(:);
Fs = opts.OutputSampleRateHz; %#ok<NASGU>
txBits = uint8(txBits(:));

metadata.numOutputSamples = numel(f_sig);
metadata.durationSeconds = numel(f_sig)/opts.OutputSampleRateHz;
metadata.rms = sqrt(mean(abs(double(f_sig)).^2));
metadata.peakMagnitude = max(abs(double(f_sig)));
metadata.matFile = relativeToRoot(matPath, opts.OutputRoot);
metadata.metadataFile = relativeToRoot(jsonPath, opts.OutputRoot);

Fs = opts.OutputSampleRateHz;
if nargin >= 8 && ~isempty(codedBits)
    codedBits = uint8(codedBits(:)); %#ok<NASGU>
    save(matPath, "f_sig", "Fs", "txBits", "codedBits", "metadata", "-v7");
else
    save(matPath, "f_sig", "Fs", "txBits", "metadata", "-v7");
end

jsonText = encodeJson(metadata);
fid = fopen(jsonPath, "w");
if fid < 0, error("Unable to open metadata path %s", jsonPath); end
cleanupObj = onCleanup(@() fclose(fid));
fwrite(fid, jsonText, "char");
clear cleanupObj;

record = struct;
record.waveformName = string(metadata.waveformName);
record.class = string(metadata.class);
record.standard = string(metadata.standard);
record.matFile = string(metadata.matFile);
record.metadataFile = string(metadata.metadataFile);
record.outputSampleRateHz = metadata.outputSampleRateHz;
record.nativeSampleRateHz = metadata.nativeSampleRateHz;
record.occupiedBandwidthHz = metadata.designedOccupiedBandwidthHz;
record.symbolRateHz = metadata.symbolRateHz;
record.modulation = string(metadata.modulation);
record.modulationSetting = string(metadata.modulationSetting);
record.pulseShape = string(metadata.pulseShape);
record.rolloff = metadata.rolloff;
record.samplesPerSymbol = metadata.samplesPerSymbol;
record.numOutputSamples = metadata.numOutputSamples;
record.durationSeconds = metadata.durationSeconds;
record.bitSource = string(metadata.bitSource);
record.payloadBitsStored = double(metadata.payloadBitsStored);
record.variationsExplored = string(metadata.variationsExplored);

records{end+1} = record;
fprintf("  %-14s %-60s (%d samples, %.3f ms)\n", metadata.class, metadata.waveformName, ...
    metadata.numOutputSamples, 1e3*metadata.durationSeconds);
end

function txt = encodeJson(s)
try
    txt = jsonencode(s, PrettyPrint=true);
catch
    txt = jsonencode(s);
end
end

function value = readStructField(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function rel = relativeToRoot(pathValue, rootValue)
p = string(pathValue); r = string(rootValue);
prefix = r + filesep;
if startsWith(p, prefix)
    rel = char(extractAfter(p, strlength(prefix)));
else
    rel = char(p);
end
end

function name = joinTags(varargin)
parts = strings(1, nargin);
for idx = 1:nargin, parts(idx) = string(varargin{idx}); end
name = sanitizeName(strjoin(parts, "_"));
end

function safe = sanitizeName(name)
safe = string(name);
safe = replace(safe, ".", "p");
safe = replace(safe, "/", "_");
safe = replace(safe, "\", "_");
safe = replace(safe, "+", "plus");
safe = replace(safe, "-", "");
safe = regexprep(safe, "[^A-Za-z0-9_]", "_");
safe = regexprep(safe, "_+", "_");
safe = regexprep(safe, "^_|_$", "");
end

function tag = formatHzTag(hz)
if hz >= 1e6
    value = hz/1e6; unit = "MHz";
elseif hz >= 1e3
    value = hz/1e3; unit = "kHz";
else
    value = hz; unit = "Hz";
end
tag = replace(string(sprintf("%.6g%s", value, unit)), ".", "p");
end

function tag = formatNumberTag(value)
tag = replace(string(sprintf("%.6g", value)), ".", "p");
end

function writeReadme(opts, manifest)
readmePath = fullfile(opts.OutputRoot, "README.md");
fid = fopen(readmePath, "w");
if fid < 0, warning("Unable to write README at %s", readmePath); return; end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, "# 245.76 MSps Waveform Library\n\n");
fprintf(fid, "Generated by `generate_waveforms_24576.m`.\n\n");
fprintf(fid, "- Output sample rate: %.2f MHz (every waveform)\n", opts.OutputSampleRateHz/1e6);
fprintf(fid, "- IQ variable in every `.mat`: `f_sig` (`complex single` = complex64 for Ettus)\n");
fprintf(fid, "- Each waveform is built at its native rate and **rationally resampled** to 245.76 MSps; `metadata.resampling.P/Q` inverts it exactly.\n");
fprintf(fid, "- Digital bit source: ITU-T PN9 (x^9 + x^5 + 1, all-ones init), stored as `txBits`.\n");
fprintf(fid, "- FM classes use `%s` as the audio source (analog, no BER).\n", opts.AudioFile);
fprintf(fid, "- Per-waveform metadata: adjacent `.json` sidecar + `metadata` struct in the `.mat`.\n");
fprintf(fid, "- Manifest: `waveform_manifest.csv` (class, occupied BW, duration, sample count, modulation, variations).\n\n");
fprintf(fid, "Decode / BER / FM-audio recovery: `decode_waveforms_24576.m`.\n\n");
fprintf(fid, "Waveform count: %d\n", height(manifest));
clear cleanupObj;
end
