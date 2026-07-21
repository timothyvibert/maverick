// Client-side helpers for dash-ag-grid (Community — no Enterprise modules).
//
// The Recent Trades grid keeps dates as sortable ISO 'YYYY-MM-DD' strings, so the
// date column filter needs a comparator that parses the string before comparing it
// to the picked date. AG-Grid's date filter calls the comparator with the picked
// date at local midnight and the cell value.
var dagfuncs = (window.dashAgGridFunctions = window.dashAgGridFunctions || {});

dagfuncs.ISODateComparator = function (filterLocalDateAtMidnight, cellValue) {
    if (!cellValue) {
        return -1;
    }
    var parts = String(cellValue).substring(0, 10).split('-');
    var cellDate = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
    if (cellDate < filterLocalDateAtMidnight) {
        return -1;
    }
    if (cellDate > filterLocalDateAtMidnight) {
        return 1;
    }
    return 0;
};

// Blotter group separators are stamped SERVER-side into row data (_group_first)
// and are only meaningful in the server's sort order. A header click re-sorts
// rows client-side while the flags stay glued to their rows, which would
// scatter separators mid-list — so the rule renders only while no column sort
// is active. Clearing the sort restores the server order and the separators.
dagfuncs.blotterGroupStart = function (params) {
    try {
        var state = params.api && params.api.getColumnState ? params.api.getColumnState() : [];
        for (var i = 0; i < state.length; i++) {
            if (state[i].sort) {
                return false;
            }
        }
    } catch (e) { /* fall through: render as the server stamped it */ }
    return !!(params.data && params.data._group_first);
};
