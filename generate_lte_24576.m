function manifest = generate_lte_24576(varargin)
%GENERATE_LTE_24576  LTE downlink RMC waveforms at 245.76 MSps (complex64 IQ).
%
% Standalone companion to generate_waveforms_24576.m: generates standard-
% compliant LTE (E-UTRA) downlink reference-measurement-channel waveforms with
% lteRMCDLTool, varying channel bandwidth (the full 1.4/3/5/10/15/20 MHz set --
% smallest to widest single-carrier LTE) and PDSCH modulation (QPSK/16QAM/64QAM),
% PN9 payload, then rationally resamples each to 245.76 MSps and stores it as a
% complex single IQ vector plus JSON metadata, alongside a new lte_manifest.csv.
%
% The transport block is sized to the PDSCH allocation at ~1/2 code rate and RV
% is fixed to 0 on every subframe (RVSeq = 0) so every subframe is an
% independently decodable first transmission -> clean-resample BER = 0.
%
%   generate_lte_24576                       % full 6 x 3 = 18 waveform set
%   generate_lte_24576("Mode","smoke")       % small subset for a quick check
%
% Decode / BER: decode_lte_24576.m

opts = parseOptions(varargin{:});
assert(exist('lteRMCDL','file')==2, ...
    'LTE Toolbox is required (lteRMCDL not found on the path).');
if ~exist(opts.OutputRoot,'dir'), mkdir(opts.OutputRoot); end

records = {};
records = generateLTEDownlink(opts, records);

if isempty(records)
    manifest = table();
else
    manifest = struct2table([records{:}]);
end
manifestPath = fullfile(opts.OutputRoot, "lte_manifest.csv");
writetable(manifest, manifestPath);
fprintf("\nWrote %d LTE waveforms\n", height(manifest));
fprintf("Manifest: %s\n", manifestPath);
end

% ======================================================================= %
function opts = parseOptions(varargin)
here = fileparts(mfilename("fullpath"));
p = inputParser;
p.addParameter("OutputRoot", fullfile(here, "generated_waveforms_24576"));
p.addParameter("OutputSampleRateHz", 245.76e6);
p.addParameter("Mode", "full");
p.addParameter("TotSubframes", 10);         % one radio frame (10 ms) <= 25 ms cap
p.addParameter("TargetCodeRate", 0.5);
p.parse(varargin{:});
opts = p.Results;
opts.Mode = string(opts.Mode);

% full single-carrier LTE bandwidth set (TS 36.101): 1.4/3/5/10/15/20 MHz
opts.NDLRB = [6 15 25 50 75 100];
opts.Mods  = ["QPSK" "16QAM" "64QAM"];
if strcmpi(opts.Mode,"smoke")
    opts.NDLRB = [6 100];                   % smallest + widest
    opts.Mods  = ["QPSK" "64QAM"];
end
end

% ======================================================================= %
function records = generateLTEDownlink(opts, records)
className = "LTE";
folder = ensureClassFolder(opts.OutputRoot, className);

for ndlrb = opts.NDLRB
    bwMHz = ndlrbToBwMHz(ndlrb);
    for modName = opts.Mods
        rmc = buildRMC(ndlrb, char(modName), opts);
        tb = rmc.PDSCH.TrBlkSizes;                     % per-subframe TB sizes
        bits = double(pn9Bits(sum(tb)));               % PN9 payload, one TB per subframe
        [nativeWave, ~, winfo] = lteRMCDLTool(rmc, bits);
        nativeWave = nativeWave(:,1);                  % single antenna port
        nativeFs = winfo.SamplingRate;

        [wave, pp, qq] = resampleToTarget(nativeWave, nativeFs, opts.OutputSampleRateHz);
        wave = normalizeWaveform(wave);
        occBw = ndlrb*12*15e3;

        name = joinTags(className, "bw"+formatHzTag(bwMHz*1e6), "pdsch"+modName, "pn9");
        metadata = baseMetadata(className, name, opts.OutputSampleRateHz);
        metadata.standard = "LTE downlink";
        metadata.bitSource = "PN9";
        metadata.payloadBitsStored = true;
        metadata.designedOccupiedBandwidthHz = occBw;
        metadata.requestedOccupiedBandwidthHz = bwMHz*1e6;
        metadata.nativeSampleRateHz = nativeFs;
        metadata.symbolRateHz = NaN;
        metadata.modulation = char(modName);
        metadata.modulationSetting = "PDSCH " + modName + ", targetCodeRate " + opts.TargetCodeRate;
        metadata.pulseShape = "CP-OFDM (SC-FDMA control)";
        metadata.rolloff = NaN;
        metadata.samplesPerSymbol = NaN;
        metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
        metadata.variationsExplored = sprintf("channelBW=%gMHz; NDLRB=%d; PDSCHmod=%s", bwMHz, ndlrb, modName);
        metadata.standardConfig = struct( ...
            "ChannelBandwidthMHz", bwMHz, "NDLRB", ndlrb, "CellRefP", rmc.CellRefP, ...
            "NCellID", rmc.NCellID, "DuplexMode", char(rmc.DuplexMode), ...
            "PDSCHModulation", char(modName), "TargetCodeRate", opts.TargetCodeRate, ...
            "TotSubframes", opts.TotSubframes, "TrBlkSizes", tb, "RVSeq", 0, ...
            "NativeSampleRateHz", nativeFs);
        metadata.decodingRecipe = "Rebuild the RMC from standardConfig (RVSeq=0), resample to NativeSampleRateHz, lteDLFrameOffset sync, lteOFDMDemodulate, per-subframe lteDLChannelEstimate + ltePDSCHDecode + lteDLSCHDecode with the stored TrBlkSizes, compare to txBits.";
        records = saveWaveformRecord(records, folder, name, wave, bits, metadata, opts);
    end
end
end

% ======================================================================= %
function rmc = buildRMC(ndlrb, modName, opts)
rmc = lteRMCDL('R.0','FDD');
rmc.NDLRB = ndlrb;
rmc.CellRefP = 1;
rmc.NCellID = 0;
rmc.PDSCH.Modulation = modName;
rmc.PDSCH.PRBSet = (0:ndlrb-1).';           % full-band allocation
rmc.PDSCH.NLayers = 1;
rmc.PDSCH.RNTI = 1;
rmc.PDSCH.TargetCodeRate = opts.TargetCodeRate;
rmc.TotSubframes = opts.TotSubframes;
% clear the inherited fixed R.0 transport-block sizes so lteRMCDL recomputes
% them for the new allocation/modulation at the target code rate
if isfield(rmc.PDSCH,'TrBlkSizes'),      rmc.PDSCH = rmfield(rmc.PDSCH,'TrBlkSizes'); end
if isfield(rmc.PDSCH,'CodedTrBlkSizes'), rmc.PDSCH = rmfield(rmc.PDSCH,'CodedTrBlkSizes'); end
rmc = lteRMCDL(rmc);
% fix RV=0 for every subframe AFTER completion (completion re-expands RVSeq to
% [0 1 2 3]); keep the default HARQ processes so all subframes still carry PDSCH
rmc.PDSCH.RVSeq = 0;
rmc.PDSCH.RV = 0;
end

function bwMHz = ndlrbToBwMHz(ndlrb)
switch ndlrb
    case 6,   bwMHz = 1.4;
    case 15,  bwMHz = 3;
    case 25,  bwMHz = 5;
    case 50,  bwMHz = 10;
    case 75,  bwMHz = 15;
    case 100, bwMHz = 20;
    otherwise, bwMHz = ndlrb*0.18;   % PRB spacing fallback
end
end

% ---- shared helpers (mirrors generate_waveforms_24576.m) ---------------- %
function [wave, p, q] = resampleToTarget(nativeWave, nativeFs, targetFs)
[p, q] = rat(targetFs/nativeFs, 1e-12);
if p == q
    wave = double(nativeWave(:));
else
    wave = resample(double(nativeWave(:)), p, q, 20);
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
metadata.createdBy = "generate_lte_24576.m";
metadata.createdOn = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss ZZZZ"));
metadata.pn9Polynomial = "x^9 + x^5 + 1";
metadata.pn9InitialState = "all ones";
end

function records = saveWaveformRecord(records, folder, waveformName, wave, txBits, metadata, opts)
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
save(matPath, "f_sig", "Fs", "txBits", "metadata", "-v7");

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

fprintf("  %-4s %-46s (%d samples, %.3f ms, occ %.2f MHz)\n", metadata.class, ...
    metadata.waveformName, metadata.numOutputSamples, 1e3*metadata.durationSeconds, ...
    metadata.designedOccupiedBandwidthHz/1e6);
end

function txt = encodeJson(s)
try
    txt = jsonencode(s, PrettyPrint=true);
catch
    txt = jsonencode(s);
end
end

function rel = relativeToRoot(pathValue, rootValue)
pathValue = char(pathValue); rootValue = char(rootValue);
if startsWith(pathValue, rootValue)
    rel = pathValue(numel(rootValue)+1:end);
    rel = regexprep(rel, "^[\\/]+", "");
else
    rel = pathValue;
end
rel = strrep(rel, "\", "/");
end

function name = joinTags(varargin)
parts = string(varargin);
parts = parts(strlength(parts) > 0);
name = strjoin(parts, "_");
end

function safe = sanitizeName(name)
safe = string(name);
safe = regexprep(safe, "[^\w\.-]", "_");
end

function tag = formatHzTag(hz)
if hz >= 1e6
    tag = formatNumberTag(hz/1e6) + "MHz";
elseif hz >= 1e3
    tag = formatNumberTag(hz/1e3) + "kHz";
else
    tag = formatNumberTag(hz) + "Hz";
end
end

function tag = formatNumberTag(value)
if value == round(value)
    tag = string(round(value));
else
    tag = string(value);
    tag = regexprep(tag, "\.", "p");
end
end
