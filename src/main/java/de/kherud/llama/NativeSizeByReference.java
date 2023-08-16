/*
 * Copyright (C) 2021 denkbares GmbH, Germany
 *
 * This is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option) any
 * later version.
 *
 * This software is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this software; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA, or see the FSF
 * site: http://www.fsf.org.
 */

package de.kherud.llama;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;

public class NativeSizeByReference extends ByReference {
    public NativeSizeByReference() {
        this(0);
    }

    public NativeSizeByReference(long value) {
        super(Native.SIZE_T_SIZE);
        setValue(value);
    }

    public void setValue(long value) {
        Pointer p = getPointer();
        switch (Native.SIZE_T_SIZE) {
            case 2:
                p.setShort(0, (short)value);
                break;
            case 4:
                p.setInt(0, (int)value);
                break;
            case 8:
                p.setLong(0, value);
                break;
            default:
                throw new RuntimeException("Unsupported size: " + Native.SIZE_T_SIZE);
        }
    }

    public long getValue() {
        Pointer p = getPointer();
        switch (Native.SIZE_T_SIZE) {
            case 2:
                return p.getShort(0) & 0xFFFFL;
            case 4:
                return p.getInt(0) & 0xFFFFFFFFL;
            case 8:
                return p.getLong(0);
            default:
                throw new RuntimeException("Unsupported size: " + Native.SIZE_T_SIZE);
        }
    }
}
