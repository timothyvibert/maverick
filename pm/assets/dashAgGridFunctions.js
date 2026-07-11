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
