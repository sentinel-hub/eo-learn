//VERSION=3

// Calculate number of bands needed for all intervals
// Initialize dates and interval
var start_date = new Date("2019-01-01");
var end_date = new Date("2020-01-01");
var sampled_dates = sample_timestamps(start_date, end_date, 10, 'day').map(d => withoutTime(d));
var nb_bands = sampled_dates.length;
var sh_dates = [];
var sum_valid = [];

function interval_search(x, arr) {
  let start_idx = 0,  end_idx = arr.length - 2;

  // Iterate while start not meets end
  while (start_idx <= end_idx) {
    // Find the mid index
    let mid_idx = (start_idx + end_idx) >> 1;

    // If element is present at mid, return True
    if (arr[mid_idx] <= x && x < arr[mid_idx + 1]) {
      return mid_idx;
    }
    // Else look in left or right half accordingly
    else if (arr[mid_idx + 1] <= x) start_idx = mid_idx + 1;
    else end_idx = mid_idx - 1;
  }
  if (x == arr[arr.length-1]){
    return arr.length-2;
  }
  return undefined;
}

function linearInterpolation(x, x0, y0, x1, y1, no_data_value = NaN) {
    if (x < x0 || x > x1) {
        return no_data_value;
    }
    var a = (y1 - y0) / (x1 - x0);
    var b = -a * x0 + y0;
    return a * x + b;
}

function lininterp(x_arr, xp_arr, fp_arr, no_data_value = NaN) {
    results = [];
    data_mask = [];
    xp_arr_idx = 0;
    for (var i = 0; i < x_arr.length; i++) {
        var x = x_arr[i];
        interval = interval_search(x, xp_arr);
        if (interval === undefined) {
            data_mask.push(0);
            results.push(no_data_value);
            continue;
        }
        data_mask.push(1);
        results.push(
            linearInterpolation(
                x,
                xp_arr[interval],
                fp_arr[interval],
                xp_arr[interval + 1],
                fp_arr[interval + 1],
                no_data_value
            )
        );
    }
    return [results, data_mask];
}

function interpolated_index(index_a, index_b) {
    // Calculates the index for all bands in array
    var index_data = [];
    for (var i = 0; i < index_a.length; i++) {
        // UINT index returned
        let ind = (index_a[i] - index_b[i]) / (index_a[i] + index_b[i]);
        index_data.push(ind * 10000 + 10000);
    }
    return index_data
}

function increase(original_date, period, period_unit) {
    date = new Date(original_date)
    switch (period_unit) {
        case 'millisecond':
            return new Date(date.setMilliseconds(date.getMilliseconds() + period));
        case 'second':
            return new Date(date.setSeconds(date.getSeconds() + period));
        case 'minute':
            return new Date(date.setMinutes(date.getMinutes() + period));
        case 'hour':
            return new Date(date.setHours(date.getHours() + period));
        case 'day':
            return new Date(date.setDate(date.getDate() + period));
        case 'month':
            return new Date(date.setMonth(date.getMonth() + period));
        default:
            return undefined
    }
}

function sample_timestamps(start, end, period, period_unit) {
    var cDate = new Date(start);
    var sampled_dates = []
    while (cDate < end) {
        sampled_dates.push(cDate);
        cDate = increase(cDate, period, period_unit);
    }
    return sampled_dates;
}

function is_valid(smp) {
    // Check if the sample is valid (i.e. contains no clouds or snow)
    let clm = smp.CLM;
    let clp = smp.CLP;
    let dm = smp.dataMask;

    if (clm != 0 || clp / 255 > 0.3 || dm != 1) {
        return false;
    }
    return true;
}

function is_valid_thr(smp, thr) {
    // Check if the sample is valid (i.e. contains no clouds or snow)
    let clp = smp.CLP;
    let dm = smp.dataMask;

    if (clp / 255 > thr || dm != 1) {
        return false;
    }
    return true;
}

function withoutTime(intime) {
    // Return date without time
    intime.setHours(0, 0, 0, 0);
    return intime;
}

function fillValues(vals_dict, sample) {
    for (var band in vals_dict) {
        vals_dict[band].push(sample[band]);
    }
}

// Sentinel Hub functions
function setup() {
    // Setup input/output parameters
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08", "B11", "B12", "CLP", "CLM", "dataMask"],
            units: "DN"
        }],
        output: [
            { id: "B02", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "B03", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "B04", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "B08", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "B11", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "B12", bands: nb_bands, sampleType: SampleType.UINT16 },
            { id: "data_mask", bands: nb_bands, sampleType: SampleType.UINT8 }
        ],
        mosaicking: "ORBIT"
    }
}

// Evaluate pixels in the bands
function evaluatePixel(samples, scenes) {
    
    // Initialise arrays
    var valid_indices = []
    var valid_dates = []
    // Loop over samples.
    for (var i = 0; i < samples.length; i++){
        if (is_valid(samples[i])) {
            valid_indices.push(i);
            valid_dates.push(withoutTime(new Date(scenes[i].date)));
        }
    }
    
    var clp_thr = 0.0;
    while (valid_indices.length < 2 || valid_dates[0] > sampled_dates[0] || valid_dates[valid_dates.length-1] < sampled_dates[sampled_dates.length-1]) {
        var valid_dates = [];
        var valid_indices = [];
        for (var i = 0; i < samples.length; i++) {
            if (is_valid_thr(samples[i], clp_thr)) {
                valid_indices.push(i);
                valid_dates.push(scenes[i].date);
            }
        }
        clp_thr += 0.05;
        if (clp_thr > 1) { break; }
    }
    
    // Fill data
    var valid_samples = { 'B02': [], 'B03': [], 'B04': [], 'B08': [], 'B11': [], 'B12': [] };
    for (var i of valid_indices) { fillValues(valid_samples, samples[i]); }
    
    // Calculate indices and return optimised for UINT16 format (will need unpacking)
    no_data_value = 10000;
    var [b02_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B02'], no_data_value);
    var [b03_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B03'], no_data_value);
    var [b04_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B04'], no_data_value);
    var [b08_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B08'], no_data_value);
    var [b11_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B11'], no_data_value);
    var [b12_interpolated, dm] = lininterp(sampled_dates, valid_dates, valid_samples['B12'], no_data_value);
    
    // Return all arrays
    return {
        B02: b02_interpolated,
        B03: b03_interpolated,
        B04: b04_interpolated,
        B08: b08_interpolated,
        B11: b11_interpolated,
        B12: b12_interpolated,
        data_mask: dm
    }
}
